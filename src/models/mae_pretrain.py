import lightning.pytorch as pl
import timm
import torch
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from torch import nn


class MAE(pl.LightningModule):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_mlp_ratio=4.0,
        decoder_proj_drop_rate=0.0,
        decoder_attn_drop_rate=0.0,
        mask_ratio=0.75,
        lr=1.5e-4,
        wd=0.05,
        b1=0.9,
        b2=0.95,
    ):
        super().__init__()
        self.save_hyperparameters()

        vit = timm.create_model(self.hparams.model_name, pretrained=True)
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=int(self.patch_size),
            embed_dim=vit.embed_dim,
            decoder_embed_dim=self.hparams.decoder_dim,
            decoder_depth=self.hparams.decoder_depth,
            decoder_num_heads=self.hparams.decoder_num_heads,
            mlp_ratio=self.hparams.decoder_mlp_ratio,
            proj_drop_rate=self.hparams.decoder_proj_drop_rate,
            attn_drop_rate=self.hparams.decoder_attn_drop_rate,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        return self.common_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.common_step("val", batch, batch_idx)

    def common_step(self, phase, batch, batch_idx):
        images, _ = batch
        batch_size = images.shape[0]

        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.hparams.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, int(self.patch_size))
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log(f"{phase}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            [param for name, param in self.named_parameters() if param.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
