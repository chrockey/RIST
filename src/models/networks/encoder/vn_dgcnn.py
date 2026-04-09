"""Vector Neuron DGCNN encoder for SO(3)-equivariant feature extraction."""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .vn_layers import (
    VNLinearLeakyReLU,
    VNStdFeature,
    build_edge_features,
    build_edge_features_cross,
    mean_pool,
)


class VNEncoder(nn.Module):
    """Vector Neuron encoder with Feature Pyramid and Theta Networks.

    Uses:
    - VN (Vector Neuron) layers for SO(3)-equivariant feature extraction
    - FPN (Feature Pyramid Network) for multi-scale feature aggregation
    - Theta network for learning local shape transform parameters

    Args:
        z_dim: Latent z dimension (default: 512)
        k: Number of nearest neighbors for graph convolution (default: 20)
        theta_dim: Dimension of local shape transform output (default: 512)
    """

    def __init__(self, z_dim: int = 512, k: int = 20, theta_dim: int = 512, use_fpn: bool = True, dynamic_knn: bool = True):
        super(VNEncoder, self).__init__()
        self.z_dim = z_dim
        self.k = k
        self.use_fpn = use_fpn
        self.dynamic_knn = dynamic_knn

        self.conv1 = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, dim=5, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3, dim=5, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3, dim=5, negative_slope=0.0)

        if use_fpn:
            fpn_dim = 256 // 3 + 128 // 3 + 64 // 3 * 2  # All 4 stages
        else:
            fpn_dim = 256 // 3  # Only final stage (feat4)
        self.conv5 = VNLinearLeakyReLU(fpn_dim, z_dim // 3, dim=4, share_nonlinearity=True)
        self.global_dim = z_dim // 3

        self.std_feature = VNStdFeature(z_dim // 3 * 2, dim=4, normalize_frame=False)

        self.latent_dim = z_dim // 3 * 2 * 3  # ~2*z_dim
        self.theta_dim = theta_dim // 3
        self.theta_net = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(self.latent_dim, self.latent_dim, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(self.latent_dim, self.theta_dim * self.global_dim, 1),
        )

        self.pool = mean_pool

    def forward(self, x: torch.Tensor):
        """Extract SO(3)-equivariant features from point cloud.

        Args:
            x: Point cloud of shape (B, N, 3)

        Returns:
            Tuple of:
            - z: Global equivariant features of shape (B, C, 3)
            - theta: Local shape transform parameters of shape (B, C, C, N)
        """
        batch_size = x.shape[0]
        n_pts = x.shape[1]

        # Edge convolution with cross product
        x = rearrange(x, 'b n d -> b 1 d n')
        x_coord = x.view(batch_size, -1, n_pts) if not self.dynamic_knn else None
        x = build_edge_features_cross(x, k=self.k)
        x = self.conv1(x)
        feat1 = self.pool(x)

        # Edge convolution (dynamic or static based on config)
        x = build_edge_features(feat1, k=self.k, x_coord=x_coord)
        x = self.conv2(x)
        feat2 = self.pool(x)

        # Edge convolution (dynamic or static based on config)
        x = build_edge_features(feat2, k=self.k, x_coord=x_coord)
        x = self.conv3(x)
        feat3 = self.pool(x)

        # Edge convolution (dynamic or static based on config)
        x = build_edge_features(feat3, k=self.k, x_coord=x_coord)
        x = self.conv4(x)
        feat4 = self.pool(x)

        # Feature pyramid (or single-scale if use_fpn=False)
        if self.use_fpn:
            x = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        else:
            x = feat4
        x = self.conv5(x)

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
