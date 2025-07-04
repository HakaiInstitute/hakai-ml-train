import math

import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, overlapping=True
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0]
        )

        if overlapping:
            # Example: Overlapping patches using a convolution with stride < kernel_size
            stride = patch_size // 2
            padding = patch_size // 2  # Or calculate precisely based on stride/kernel
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=padding,
            )
            # Calculate num_patches for overlapping case (more complex, depends on padding/stride)
            # For simplicity in this example, we'll approximate or assume it's handled later dynamically
            # A more robust way is to calculate output feature map size:
            h_out = math.floor(
                (self.img_size[0] + 2 * padding - patch_size) / stride + 1
            )
            w_out = math.floor(
                (self.img_size[1] + 2 * padding - patch_size) / stride + 1
            )
            self.num_patches = h_out * w_out
            self.grid_size = (h_out, w_out)  # Store grid size for later reshaping

        else:
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.grid_size = (
                self.img_size[0] // patch_size,
                self.img_size[1] // patch_size,
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.norm = nn.LayerNorm(embed_dim)  # Optional LayerNorm after projection

    def forward(self, x):
        B, C, H, W = x.shape
        assert self.img_size[0] == H and self.img_size[1] == W, (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )

        x = self.proj(x)  # Shape: B, E, H/P, W/P (or different if overlapping)
        # Store feature map size for decoder
        self.feature_map_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # Shape: B, N, E (where N = H/P * W/P)
        x = self.norm(x)  # Apply LayerNorm
        return x


class MLP(nn.Module):
    """Simple Feed Forward Network"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Encoder Block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Apply norm before attention (Pre-Norm)
        x = x + self.mlp(self.norm2(x))  # Apply norm before MLP (Pre-Norm)
        return x
