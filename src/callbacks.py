import lightning.pytorch as pl


class MaskAnnealingCallback(pl.Callback):
    """Linearly decay attn_mask_probs from 1.0 to 0.0 over training.

    Implements the mask annealing scheme from the EoMT paper: masked attention
    is applied with probability P_mask that starts at 1.0 and linearly decays
    to 0.0 over the course of training. This allows the model to benefit from
    masked attention early on while learning to operate without it, enabling
    efficient inference with a plain ViT architecture.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        progress = trainer.global_step / max(1, trainer.estimated_stepping_batches)
        prob = max(0.0, 1.0 - progress)

        # Handle torch.compile wrapper
        model = pl_module.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        model.attn_mask_probs.fill_(prob)

        pl_module.log("mask_anneal/prob", prob)
