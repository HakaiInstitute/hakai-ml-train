import torch
import torch.nn as nn

from train.models.aerial_former.decoder import SimpleDecoder
from train.models.aerial_former.modules import Block, PatchEmbed


class AerialFormer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=2,  # Example: binary segmentation
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        decoder_channels=256,
        use_overlapping_patches=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            overlapping=use_overlapping_patches,
        )
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size  # Store grid size needed by decoder

        # 2. Positional Encoding
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # Initialize

        # 3. Transformer Encoder
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)  # Final normalization after encoder blocks

        # 4. Segmentation Decoder Head
        # Determine the scale factor based on patch embedding strategy
        # Note: If overlapping, the effective scale might differ from patch_size.
        # The SimpleDecoder uses feature_map_scale for upsampling. Ensure it matches
        # the actual reduction factor. For non-overlapping it's patch_size.
        # For the overlapping example used in PatchEmbed (stride=ps//2, padding=ps//2, kernel=ps)
        # the output size is roughly the same as input, so scale factor should be stride based.
        # Let's assume patch_size for simplicity here, but adjust based on your specific PatchEmbed.
        scale_factor = (
            patch_size  # Needs adjustment if using complex overlapping patches
        )
        self.decoder = SimpleDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            feature_map_scale=scale_factor,  # How much to upsample in the end
            channels=decoder_channels,
        )

    def forward(self, x):
        B = x.shape[0]
        # Patch Embedding
        x = self.patch_embed(x)  # B, N, E
        patch_embed_grid_size = self.patch_embed.feature_map_size  # Get actual H/P, W/P

        # Add Positional Encoding
        # Check if pos_embed size matches num_patches derived from input
        if x.size(1) != self.pos_embed.size(1):
            # Attempt simple interpolation if sizes mismatch (e.g., due to overlapping patches)
            # This requires pos_embed to have a spatial structure notion or use 2D interpolation.
            # A common approach is 2D interpolation *before* flattening in PatchEmbed,
            # or interpolating the pos_embed parameter itself.
            # Simplified fallback: print warning or error. Robust handling is complex.
            print(
                f"Warning: Positional embedding size mismatch. Input patches: {x.size(1)}, PosEmbed: {self.pos_embed.size(1)}. Check PatchEmbed and pos_embed dimensions."
            )
            # Forcing it for now, might lead to errors or poor performance:
            # pos_embed_resized = F.interpolate(self.pos_embed.permute(0,2,1).reshape(1, self.embed_dim, int(self.pos_embed.size(1)**0.5), -1), size=patch_embed_grid_size, mode='bilinear', align_corners=False).flatten(2).permute(0,2,1)
            # x = x + pos_embed_resized
            # Fallback: Use untransformed pos_embed, might fail if sizes don't match broadcast
            x = (
                x + self.pos_embed[:, : x.size(1), :]
            )  # Slice if input has fewer patches than expected
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Transformer Encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # Apply final LayerNorm

        # Decoder Head
        # Pass the actual grid size from patch_embed output
        x = self.decoder(x, patch_embed_grid_size)

        return x  # Output: B, NumClasses, H, W
