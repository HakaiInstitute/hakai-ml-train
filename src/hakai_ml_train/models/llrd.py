"""Layer-wise learning rate decay (LLRD) for ViT-style encoders.

LLRD assigns progressively smaller learning rates to earlier transformer
layers, preserving low-level pretrained features while letting upper layers
adapt to the downstream task. This is the standard recipe used by BEiT, MAE,
and DINO for ViT-Large fine-tuning.
"""

from __future__ import annotations

import torch.nn as nn

_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _resolve(obj, dotted: str):
    """Walk `dotted` (e.g., 'model.encoder') from `obj`."""
    for attr in dotted.split("."):
        obj = getattr(obj, attr)
    return obj


def _no_decay_ids(root: nn.Module) -> set[int]:
    """Return id()s of params that should use weight_decay=0 (norm + bias)."""
    no_wd = set()
    for mod in root.modules():
        if isinstance(mod, _NORM_TYPES):
            no_wd.update(id(p) for p in mod.parameters(recurse=False))
    for name, p in root.named_parameters():
        if name.endswith(".bias"):
            no_wd.add(id(p))
    return no_wd


def _layer_id_for_encoder_param(name: str, num_layers: int) -> int:
    """Map an encoder parameter name to a layer id.

    Lower id = earlier layer = smaller LR.

    - patch_embed.*, pos_embed, cls_token, wavelength encoder -> 0
    - blocks.{i}.*                                            -> i + 1
    - norm.*  (final ViT norm)                                -> num_layers + 1
    - anything else inside the encoder wrapper                -> num_layers + 1
    """
    parts = name.split(".")
    if (
        parts[0] in ("patch_embed", "pos_embed", "cls_token")
        or "wave" in parts[0].lower()
    ):
        return 0
    if parts[0] == "blocks":
        # parts: ["blocks", "<i>", ...]
        return int(parts[1]) + 1
    # norm, head, or anything else -> top
    return num_layers + 1


def build_llrd_param_groups(
    module,
    decay_rate: float,
    encoder_attr: str = "model.encoder",
    vit_attr: str = "dofa_model",
    num_layers: int | None = None,
) -> list[dict]:
    """Build LLRD parameter groups for a Lightning module.

    Args:
        module: LightningModule with the model attached.
        decay_rate: per-layer decay (e.g., 0.75). lr_scale = decay_rate**(top - layer_id).
        encoder_attr: dotted path from `module` to the encoder wrapper.
        vit_attr: attribute on the encoder wrapper holding the raw ViT (with .blocks).
            Pass an empty string if the encoder IS the ViT.
        num_layers: number of transformer blocks. Auto-detected from `len(vit.blocks)`
            if None.

    Returns:
        List of optimizer param-group dicts. Each has:
            - params: list[nn.Parameter]
            - lr_scale: float
            - weight_decay: 0.0  (only on no-decay groups; otherwise absent)
            - name: str
    """
    encoder = _resolve(module, encoder_attr)
    vit = _resolve(encoder, vit_attr) if vit_attr else encoder

    if num_layers is None:
        num_layers = len(vit.blocks)

    top_layer = num_layers + 1
    no_wd_ids = _no_decay_ids(module)

    # Iterate all named params on the full module so decoder/head land in the
    # "top" group automatically.
    buckets: dict[tuple[int, bool], list] = {}
    seen: set[int] = set()

    # First, encoder params with proper layer ids based on their name within the ViT
    for name, p in vit.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        layer_id = _layer_id_for_encoder_param(name, num_layers)
        has_decay = id(p) not in no_wd_ids
        buckets.setdefault((layer_id, has_decay), []).append(p)
        seen.add(id(p))

    # Encoder params NOT inside the ViT (e.g., wavelength encoder on the wrapper)
    for _, p in encoder.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        # Treat extra encoder-level params as the earliest layer
        layer_id = 0
        has_decay = id(p) not in no_wd_ids
        buckets.setdefault((layer_id, has_decay), []).append(p)
        seen.add(id(p))

    # Everything else (decoder, head, necks) -> top layer
    for _, p in module.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        layer_id = top_layer
        has_decay = id(p) not in no_wd_ids
        buckets.setdefault((layer_id, has_decay), []).append(p)
        seen.add(id(p))

    groups: list[dict] = []
    for (layer_id, has_decay), params in sorted(buckets.items()):
        if not params:
            continue
        lr_scale = decay_rate ** (top_layer - layer_id)
        tag = "decay" if has_decay else "no_decay"
        group: dict = {
            "params": params,
            "lr_scale": lr_scale,
            "name": f"layer_{layer_id}.{tag}",
        }
        if not has_decay:
            group["weight_decay"] = 0.0
        groups.append(group)
    return groups


if __name__ == "__main__":
    # Smoke test against the real DOFA Large model.

    from src.models.terratorch import TerraTorchSegmentationModel

    model = TerraTorchSegmentationModel(
        loss="LabelSmoothingLovasz",
        loss_opts=dict(mode="binary", ignore_index=-100),
        num_classes=1,
        ignore_index=-100,
        optimizer_class="torch.optim.AdamW",
        optimizer_opts=dict(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.99]),
        lr_scheduler_class="torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts=dict(max_lr=1e-4, pct_start=0.3),
        lr_scheduler_interval="step",
        model_opts=dict(
            backbone="dofa_large_patch16_224_custom",
            backbone_pretrained=False,  # avoid network fetch in smoke test
            backbone_wavelengths=[
                0.442,
                0.49,
                0.531,
                0.565,
                0.610,
                0.665,
                0.705,
                0.865,
            ],
            backbone_out_indices=[5, 11, 17, 23],
            necks=[
                dict(name="ReshapeTokensToImage", remove_cls_token=True),
                dict(name="LearnedInterpolateToPyramidal"),
            ],
            decoder="UperNetDecoder",
            decoder_channels=512,
            head_channel_list=[512],
            head_dropout=0.1,
        ),
    )

    groups = build_llrd_param_groups(
        model,
        decay_rate=0.75,
        encoder_attr="model.encoder",
        vit_attr="dofa_model",
    )

    # All trainable params accounted for exactly once
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    assert len(grouped_ids) == len(all_trainable), (
        f"missing params: {len(all_trainable) - len(grouped_ids)}"
    )

    # Layer ids: 0..24 plus top (25)
    layer_ids = sorted(
        {int(g["name"].split(".")[0].removeprefix("layer_")) for g in groups}
    )
    assert layer_ids[0] == 0 and layer_ids[-1] == 25, layer_ids

    # lr_scale monotonically non-decreasing with layer id
    scales_by_layer: dict[int, float] = {}
    for g in groups:
        lid = int(g["name"].split(".")[0].removeprefix("layer_"))
        scales_by_layer[lid] = g["lr_scale"]
    prev = -1.0
    for lid in sorted(scales_by_layer):
        assert scales_by_layer[lid] >= prev, (lid, scales_by_layer)
        prev = scales_by_layer[lid]

    # Top group is exactly 1.0
    assert abs(scales_by_layer[25] - 1.0) < 1e-9, scales_by_layer[25]

    # Spot check decay formula
    expected_block0 = 0.75 ** (25 - 1)
    assert abs(scales_by_layer[1] - expected_block0) < 1e-9, (
        scales_by_layer[1],
        expected_block0,
    )

    # Norm/bias params land in no_decay groups with weight_decay=0.0
    no_decay_groups = [g for g in groups if g["name"].endswith(".no_decay")]
    assert no_decay_groups, "expected at least one no_decay group"
    for g in no_decay_groups:
        assert g.get("weight_decay") == 0.0, g["name"]
    decay_groups = [g for g in groups if g["name"].endswith(".decay")]
    assert decay_groups, "expected at least one decay group"
    for g in decay_groups:
        assert "weight_decay" not in g, g["name"]

    print(f"OK: {len(groups)} groups, {len(all_trainable)} trainable params")
    for g in groups:
        print(f"  {g['name']:30s} lr_scale={g['lr_scale']:.6f}  n={len(g['params'])}")

    # freeze_backbone=True collapses LLRD to just the top (decoder/head) groups
    model_frozen = TerraTorchSegmentationModel(
        loss="LabelSmoothingLovasz",
        loss_opts=dict(mode="binary", ignore_index=-100),
        num_classes=1,
        ignore_index=-100,
        optimizer_class="torch.optim.AdamW",
        optimizer_opts=dict(lr=1e-4, weight_decay=0.01, betas=[0.9, 0.99]),
        lr_scheduler_class="torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts=dict(max_lr=1e-4, pct_start=0.3),
        lr_scheduler_interval="step",
        freeze_backbone=True,
        model_opts=dict(
            backbone="dofa_large_patch16_224_custom",
            backbone_pretrained=False,
            backbone_wavelengths=[
                0.442,
                0.49,
                0.531,
                0.565,
                0.610,
                0.665,
                0.705,
                0.865,
            ],
            backbone_out_indices=[5, 11, 17, 23],
            necks=[
                dict(name="ReshapeTokensToImage", remove_cls_token=True),
                dict(name="LearnedInterpolateToPyramidal"),
            ],
            decoder="UperNetDecoder",
            decoder_channels=512,
            head_channel_list=[512],
            head_dropout=0.1,
        ),
    )
    frozen_groups = build_llrd_param_groups(
        model_frozen,
        decay_rate=0.75,
        encoder_attr="model.encoder",
        vit_attr="dofa_model",
    )
    frozen_layer_ids = {
        int(g["name"].split(".")[0].removeprefix("layer_")) for g in frozen_groups
    }
    assert frozen_layer_ids == {25}, frozen_layer_ids
    print(f"OK frozen: {len(frozen_groups)} groups (decoder/head only)")
