import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, feature_map_scale=16, channels=256):
        """
        Args:
            embed_dim (int): Dimension of the transformer output embeddings.
            num_classes (int): Number of segmentation classes.
            feature_map_scale (int): The factor by which the spatial resolution was reduced
                                     by patch embedding (e.g., patch_size if non-overlapping).
            channels (int): Number of channels in intermediate conv layers.
        """
        super().__init__()
        self.scale_factor = feature_map_scale
        # Project embeddings back to a lower dimension suitable for convs
        self.proj = nn.Conv2d(embed_dim, channels, kernel_size=1)
        # Convolutional layers (could be more complex)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Final classification layer
        self.final_conv = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x, grid_size):
        """
        Args:
            x (Tensor): Output from transformer encoder (B, N, E).
            grid_size (tuple): Spatial dimensions of the feature map before flattening (H/P, W/P).
        """
        B, N, E = x.shape
        H_feat, W_feat = grid_size
        # Reshape sequence back into spatial grid: B, N, E -> B, E, H_feat, W_feat
        x = x.transpose(1, 2).reshape(B, E, H_feat, W_feat)

        # Process through decoder layers
        x = self.proj(x)  # B, C_dec, H_feat, W_feat
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Upsample to original image size
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

        # Final classification
        x = self.final_conv(x)  # B, NumClasses, H, W
        return x
