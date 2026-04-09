"""Simple Vector Neuron encoder for SO(3)-equivariant feature extraction.

Adapted from VNTSimpleEncoder, removing translation-invariance components
to focus on rotation equivariance only.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .vn_layers import (
    VNLinear,
    VNLinearLeakyReLU,
    VNBatchNorm,
    VNStdFeature,
    build_edge_features_cross,
    mean_pool,
)


class VNSimpleEncoder(nn.Module):
    """Simple Vector Neuron encoder with SO(3) equivariance.

    A simpler alternative to VNEncoder with fewer layers and no feature pyramid.
    Uses a single-scale architecture with optional feature transform.

    Args:
        z_dim: Latent z dimension (default: 512)
        k: Number of nearest neighbors for graph convolution (default: 20)
        theta_dim: Dimension of local shape transform output (default: 256)
        base_ch: Base channel dimension (default: 64)
        feature_transform: Whether to use global feature transform (default: False)
    """

    def __init__(
        self,
        z_dim: int = 512,
        k: int = 20,
        theta_dim: int = 256,
        base_ch: int = 64,
        feature_transform: bool = False,
        **kwargs,  # Ignore extra params (e.g., use_fpn, dynamic_knn from sweep)
    ):
        super(VNSimpleEncoder, self).__init__()
        self.z_dim = z_dim
        self.k = k
        self.feature_transform = feature_transform

        # Initial convolution on edge features (3 channels from cross product)
        self.conv_pos = VNLinearLeakyReLU(3, base_ch // 3, dim=5, negative_slope=0.0)

        # Main convolution path
        self.conv1 = VNLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0)

        if self.feature_transform:
            # Feature transform network (STN-like)
            self.fstn = VNFeatureTransform(base_ch // 3)
            self.conv2 = VNLinearLeakyReLU(base_ch // 3 * 2, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        else:
            self.conv2 = VNLinearLeakyReLU(base_ch // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)

        # Output convolution
        output_ch = z_dim // 3
        self.conv3 = VNLinear((2 * base_ch) // 3, output_ch)
        self.bn3 = VNBatchNorm(output_ch, dim=4)

        self.global_dim = z_dim // 3
        self.pool = mean_pool

        # Standard frame feature extractor for theta computation
        self.std_feature = VNStdFeature(z_dim // 3 * 2, dim=4, normalize_frame=False)

        # Theta network: maps invariant features to local shape transforms
        self.latent_dim = z_dim // 3 * 2 * 3  # ~2*z_dim
        self.theta_dim = theta_dim // 3
        self.theta_net = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(self.latent_dim, self.latent_dim, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(self.latent_dim, self.theta_dim * self.global_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """Extract SO(3)-equivariant features from point cloud.

        Args:
            x: Point cloud of shape (B, N, 3)

        Returns:
            Tuple of:
            - z: Global equivariant features of shape (B, C, 3)
            - theta: Local shape transform parameters of shape (B, C, C, N)
        """
        n_pts = x.shape[1]

        # Build edge features with cross product
        x = rearrange(x, 'b n d -> b 1 d n')
        feat = build_edge_features_cross(x, k=self.k)

        # Initial convolution and pooling
        x = self.conv_pos(feat)
        x = self.pool(x)

        # Main convolution path
        x = self.conv1(x)

        if self.feature_transform:
            # Global feature transform
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, n_pts)
            x = torch.cat((x, x_global), dim=1)

        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        # SO(3)-equivariant -> SO(3)-invariant
        z = mean_pool(x)
        x_mean = repeat(z, 'b c d -> b c d n', n=n_pts)
        x = torch.cat((x, x_mean), dim=1)
        x, _ = self.std_feature(x)
        x = rearrange(x, 'b c d n -> b (c d) n')

        # Theta network: local shape transform parameters
        theta = self.theta_net(x)
        theta = rearrange(theta, 'b (c d) n -> b c d n', c=self.theta_dim)

        return z, theta


class VNFeatureTransform(nn.Module):
    """Vector Neuron feature transform network (STN-like).

    Learns a global feature transformation for the point cloud.
    """

    def __init__(self, in_channels: int):
        super(VNFeatureTransform, self).__init__()
        self.in_channels = in_channels

        self.conv1 = VNLinearLeakyReLU(in_channels, in_channels, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(in_channels, in_channels, dim=4, negative_slope=0.0)

        self.pool = mean_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform features.

        Args:
            x: Point features of shape (B, C, 3, N)

        Returns:
            Global transformed features of shape (B, C, 3)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)  # (B, C, 3)
        return x
