"""Export a trained model to a CPU-optimised ONNX graph.

Takes the same `(config, ckpt)` pair as `kom_onnx`, and applies the three things
that actually move CPU latency:

1. A batch dim fixed at 1, which measured ~10% faster than a symbolic one and
   costs nothing given CPU inference tiles one chip at a time. Tile size stays
   dynamic, which measured free, so one artifact serves any tile size;
   `--no-dynamic-spatial` pins it but is rarely worth it.
2. Shape inference + graph pre-processing, so the quantizer sees a clean graph.
3. Static int8 QDQ quantization, calibrated on real chips drawn from the
   config's own `data` section (using its `test_transforms`, so calibration
   sees exactly the distribution inference will).

Both the fp32 and int8 graphs are written out, then benchmarked against each
other for latency and prediction agreement so the accuracy cost is visible
before anything ships.

Usage:
    python -m hakai_ml_train.deploy.kom_onnx_quantize \
        configs/kelp-rgb/segformer_resnet34.yaml \
        checkpoints/best.ckpt \
        models/kelp_rgb_segformer_resnet34 \
        --image-size 512 --calib-samples 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import albumentations as A
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import CalibrationDataReader, quantize_static
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.export import Dim

from hakai_ml_train.deploy.kom_onnx import (
    ONNXModel,
    encoder_min_image_size,
    load_model_from_config,
)


def _build_calibration_transform(config: dict, image_size: int) -> A.Compose:
    """Reuse the config's test_transforms, forced to a fixed `image_size` crop.

    Chips on disk are usually larger than the export size, so a centre crop is
    prepended. Normalize/ToTensorV2 come from the config itself, which keeps
    calibration statistics consistent with what the model saw in validation.
    """
    test_transforms = config["data"]["init_args"].get("test_transforms")
    if test_transforms is None:
        raise ValueError(
            "Config has no data.init_args.test_transforms; cannot build a "
            "calibration pipeline that matches inference preprocessing."
        )
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0.0,
                p=1.0,
            ),
            A.CenterCrop(height=image_size, width=image_size, p=1.0),
            A.from_dict(test_transforms),
        ]
    )


def _calibration_chip_dir(config: dict, split: str) -> Path:
    """Resolve the chip directory for `split` from the config's data section."""
    data_args = config["data"]["init_args"]
    key = f"{split}_chip_dir"
    if key not in data_args:
        raise ValueError(f"Config data.init_args has no '{key}'")
    return Path(data_args[key])


class NpzCalibrationDataReader(CalibrationDataReader):
    """Feeds real chips to the static quantizer's calibration pass."""

    def __init__(
        self,
        chip_dir: Path,
        transform: A.Compose,
        input_name: str,
        num_samples: int,
        seed: int = 42,
    ):
        chips = sorted(chip_dir.glob("*.npz"))
        if not chips:
            raise ValueError(f"No .npz chips found in {chip_dir}")
        if len(chips) > num_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(chips), size=num_samples, replace=False)
            chips = [chips[i] for i in sorted(idx)]

        self.chips = chips
        self.transform = transform
        self.input_name = input_name
        self._iter = None

    def _load(self, path: Path) -> np.ndarray:
        data = np.load(path)
        augmented = self.transform(image=data["image"], mask=data["label"])
        image = augmented["image"]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        return image[None].astype(np.float32)

    def get_next(self):
        if self._iter is None:
            self._iter = iter(self.chips)
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: self._load(path)}

    def rewind(self):
        self._iter = None


def export_for_quantization(
    model: torch.nn.Module,
    output_path: Path,
    num_channels: int,
    image_size: int,
    dynamic_spatial: bool,
    min_image_size: int,
) -> None:
    """Export at batch 1, optionally keeping tile size dynamic.

    Batch is pinned: a symbolic batch dim measured ~10% slower at batch 1, 512px
    on Segformer/resnet34 (interleaved medians, consistent across min and p90),
    and batching buys under 5% throughput on CPU because it is already compute
    saturated. Spatial dims stay dynamic by default at no measurable cost, so one
    artifact serves any tile size.
    """
    onnx_model = ONNXModel(model)
    onnx_model.eval()

    x = torch.rand(1, num_channels, image_size, image_size, requires_grad=False)
    if dynamic_spatial:
        spatial = (
            Dim("height", min=min_image_size),
            Dim("width", min=min_image_size),
        )
    else:
        spatial = (Dim.STATIC, Dim.STATIC)
    dynamic_shapes = {"x": (Dim.STATIC, Dim.STATIC, *spatial)}

    with torch.no_grad():
        torch.onnx.export(
            onnx_model,
            (x,),
            output_path,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            # Keep weights in the .onnx file; a sidecar .data would have to be
            # distributed alongside it, and the int8 output is self-contained.
            external_data=False,
            verbose=False,
        )


def _pre_process(src: Path, dst: Path) -> None:
    """Shape-infer and fold the graph so the quantizer sees clean, typed tensors.

    ORT's symbolic shape inference asserts on some graphs that combine any
    symbolic dim with MatMul (the Segformer all-MLP decoder is one), so fall
    back to skipping it. Quantization still works; only some fusions are lost,
    and in practice the int8 graph is no slower for it.
    """
    try:
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=False)
    except Exception as e:  # noqa: BLE001 - any failure here is non-fatal
        print(
            f"      symbolic shape inference failed ({type(e).__name__}); retrying without it"
        )
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=True)


def _model_size_mb(model_path: Path) -> float:
    """Total on-disk size, including any external-data sidecar."""
    total = model_path.stat().st_size
    sidecar = model_path.with_suffix(model_path.suffix + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total / 1e6


def _input_name(model_path: Path) -> str:
    return onnx.load(model_path).graph.input[0].name


def _make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), opts, providers=["CPUExecutionProvider"]
    )


def benchmark_and_compare(
    fp32_path: Path,
    int8_path: Path,
    chip_dir: Path,
    transform: A.Compose,
    image_size: int,
    threads: int,
    num_samples: int = 16,
) -> None:
    """Report latency for both graphs and how often their predictions agree."""
    fp32_sess = _make_session(fp32_path, threads)
    int8_sess = _make_session(int8_path, threads)
    fp32_in = fp32_sess.get_inputs()[0].name
    int8_in = int8_sess.get_inputs()[0].name

    chips = sorted(chip_dir.glob("*.npz"))[:num_samples]
    batches = []
    for path in chips:
        data = np.load(path)
        image = transform(image=data["image"], mask=data["label"])["image"]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        batches.append(image[None].astype(np.float32))

    def _time(sess, name):
        sess.run(None, {name: batches[0]})  # warmup
        t = time.perf_counter()
        for b in batches:
            sess.run(None, {name: b})
        return (time.perf_counter() - t) / len(batches)

    fp32_dt = _time(fp32_sess, fp32_in)
    int8_dt = _time(int8_sess, int8_in)

    agree, total = 0, 0
    for b in batches:
        a = np.argmax(np.asarray(fp32_sess.run(None, {fp32_in: b})[0]), axis=1)
        c = np.argmax(np.asarray(int8_sess.run(None, {int8_in: b})[0]), axis=1)
        agree += int((a == c).sum())
        total += a.size

    print()
    print(f"Benchmark on {len(batches)} chips @ {image_size}px, {threads} threads")
    print(
        f"  fp32 : {fp32_dt * 1000:8.1f} ms/tile   ({_model_size_mb(fp32_path):.1f} MB)"
    )
    print(
        f"  int8 : {int8_dt * 1000:8.1f} ms/tile   ({_model_size_mb(int8_path):.1f} MB)"
    )
    print(f"  speedup: {fp32_dt / int8_dt:.2f}x")
    print(f"  pixel agreement fp32 vs int8: {100 * agree / total:.2f}%")
    print()
    print(
        "Agreement is a smoke test, not an accuracy measure. Run `trainer.py test` "
        "against the int8 graph before shipping."
    )


def main(
    config_path: Path,
    ckpt_path: Path,
    output_prefix: Path,
    image_size: int,
    calib_split: str,
    calib_samples: int,
    threads: int,
    dynamic_spatial: bool,
    min_image_size: int | None,
    per_channel: bool,
    skip_benchmark: bool,
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model, init_args = load_model_from_config(config_path, ckpt_path)
    num_channels = init_args.get("model_opts", {}).get("in_channels", 3)
    if min_image_size is None:
        min_image_size = encoder_min_image_size(model)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = output_prefix.with_name(f"{output_prefix.name}_fp32.onnx")
    prepped_path = output_prefix.with_name(f"{output_prefix.name}_fp32_prepped.onnx")
    int8_path = output_prefix.with_name(f"{output_prefix.name}_int8.onnx")

    shape_desc = (
        f"1x{num_channels}xHxW (H,W dynamic, min {min_image_size}px)"
        if dynamic_spatial
        else f"1x{num_channels}x{image_size}x{image_size}"
    )
    print(f"[1/4] Exporting fp32 graph, input {shape_desc}...")
    export_for_quantization(
        model, fp32_path, num_channels, image_size, dynamic_spatial, min_image_size
    )
    print(f"      wrote {fp32_path}")

    print("[2/4] Running shape inference and graph pre-processing...")
    _pre_process(fp32_path, prepped_path)

    print(
        f"[3/4] Calibrating on {calib_samples} chips from the '{calib_split}' split..."
    )
    transform = _build_calibration_transform(config, image_size)
    chip_dir = _calibration_chip_dir(config, calib_split)
    reader = NpzCalibrationDataReader(
        chip_dir, transform, _input_name(prepped_path), calib_samples
    )

    quantize_static(
        str(prepped_path),
        str(int8_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
    )
    prepped_path.unlink()
    print(f"      wrote {int8_path}")

    print("[4/4] Benchmarking...")
    if skip_benchmark:
        print("      skipped")
        return
    benchmark_and_compare(
        fp32_path, int8_path, chip_dir, transform, image_size, threads
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="Path to config YAML file")
    parser.add_argument(
        "ckpt_path", type=Path, help="Path to PyTorch Lightning checkpoint"
    )
    parser.add_argument(
        "output_prefix",
        type=Path,
        help="Output path prefix; _fp32.onnx and _int8.onnx are appended",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help=(
            "Tile size used for calibration and benchmarking, and the fixed "
            "export size under --no-dynamic-spatial (default: 512)"
        ),
    )
    parser.add_argument(
        "--calib-split",
        default="val",
        choices=["train", "val", "test"],
        help="Which chip dir to calibrate on (default: val)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=256,
        help="Number of chips to calibrate on (default: 256)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="intra_op_num_threads used for benchmarking (default: 4)",
    )
    parser.add_argument(
        "--no-dynamic-spatial",
        action="store_true",
        help=(
            "Pin H/W to --image-size. Batch is always fixed at 1. Costs "
            "flexibility but lets ORT fuse more aggressively"
        ),
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=None,
        help=(
            "Smallest tile the dynamic graph accepts "
            "(default: the encoder's output_stride)"
        ),
    )
    parser.add_argument(
        "--no-per-channel",
        action="store_true",
        help="Use per-tensor weight quantization instead of per-channel",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the fp32-vs-int8 latency and agreement comparison",
    )

    args = parser.parse_args()

    main(
        args.config_path,
        args.ckpt_path,
        args.output_prefix,
        image_size=args.image_size,
        calib_split=args.calib_split,
        calib_samples=args.calib_samples,
        threads=args.threads,
        dynamic_spatial=not args.no_dynamic_spatial,
        min_image_size=args.min_image_size,
        per_channel=not args.no_per_channel,
        skip_benchmark=args.skip_benchmark,
    )
