"""Implicit function decoder for point cloud reconstruction."""

import torch
from torch import nn
from einops import repeat, einsum, rearrange

from src.models.networks.encoder.vn_layers import VNLinearLeakyReLU, VNLinear


class ImplicitDecoder(nn.Module):
    """SO(3)-equivariant implicit function decoder.

    Reconstructs point clouds from global equivariant features and
    local shape transform parameters.

    Args:
        theta_dim: Dimension of local shape transform (will be divided by 3 internally)
    """

    def __init__(self, theta_dim: int = 256):
        super(ImplicitDecoder, self).__init__()
        latent_dim = theta_dim // 3
        self.mlp = VectorMLP(latent_dim=latent_dim)

    def forward(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Decode point cloud from features.

        Args:
            z: SO(3)-equivariant global feature of shape (B, C, 3)
            theta: Local shape transform parameters of shape (B, C, C, N)

        Returns:
            Reconstructed point cloud of shape (B, N, 3)
        """
        n_pts = theta.shape[-1]

        z = repeat(z, 'b c d -> b c d n', n=n_pts)
        x = einsum(theta, z, 'b i j n, b j d n -> b i d n')
        x = self.mlp(x, format_bnc=True)
        return x


class VectorMLP(nn.Module):
    """Multi-layer perceptron using Vector Neuron layers.

    Processes 3D vector features while maintaining SO(3) equivariance.

    Args:
        latent_dim: Latent feature dimension (default: 1024)
    """

    def __init__(self, latent_dim: int = 1024):
        super(VectorMLP, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = VNLinearLeakyReLU(self.latent_dim, self.latent_dim, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(self.latent_dim, self.latent_dim // 2, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(self.latent_dim // 2, self.latent_dim // 4, dim=4, negative_slope=0.0)
        self.conv4 = VNLinear(self.latent_dim // 4, 1)

    def forward(self, x: torch.Tensor, format_bnc: bool = False) -> torch.Tensor:
        """Process vector features through MLP.

        Args:
            x: Input features of shape (B, C, 3, N)
            format_bnc: If True, return shape (B, N, 3) instead of (B, 1, 3, N)

        Returns:
            Processed features
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if format_bnc:
            x = rearrange(x, 'b 1 d n -> b n d')

        return x
