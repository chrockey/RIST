"""Vector Neuron (VN) layers for SO(3)-equivariant neural networks.

VN layers operate on 3D vector features, preserving equivariance to rotations.
Each "neuron" is a 3D vector rather than a scalar, enabling geometric reasoning.

References:
    Deng et al., "Vector Neurons: A General Framework for SO(3)-Equivariant Networks", ICCV 2021
"""

import torch
import torch.nn as nn

EPS = 1e-6


def compute_knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbor indices.

    Args:
        x: Point features of shape (B, C, N)
        k: Number of nearest neighbors

    Returns:
        Neighbor indices of shape (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_dist = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_dist.topk(k=k, dim=-1)[1]
    return idx


def build_edge_features(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None, x_coord: torch.Tensor = None) -> torch.Tensor:
    """Build edge features for graph convolution.

    Args:
        x: Point features of shape (B, C, 3, N)
        k: Number of nearest neighbors
        idx: Pre-computed neighbor indices (optional)
        x_coord: Coordinates for knn computation (optional)

    Returns:
        Edge features of shape (B, 2*C, 3, N, k)
    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if x_coord is None:
            idx = compute_knn(x, k=k)
        else:
            idx = compute_knn(x_coord, k=k)

    device = x.device
    batch_offset = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + batch_offset
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


def build_edge_features_cross(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None, return_idx: bool = False):
    """Build edge features with cross product for graph convolution.

    Args:
        x: Point features of shape (B, C, 3, N)
        k: Number of nearest neighbors
        idx: Pre-computed neighbor indices (optional)
        return_idx: Whether to return neighbor indices

    Returns:
        Edge features of shape (B, 3*C, 3, N, k), optionally with indices
    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = compute_knn(x, k=k)

    device = x.device
    batch_offset = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    knn_idx = idx + batch_offset
    knn_idx = knn_idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[knn_idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    if return_idx:
        return feature, idx
    else:
        return feature


class VNLinear(nn.Module):
    """Vector Neuron linear layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point features of shape [B, N_feat, 3, N_samples, ...]
        """
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLinearLeakyReLU(nn.Module):
    """Vector Neuron linear layer with batch norm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2
    ):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point features of shape [B, N_feat, 3, N_samples, ...]
        """
        # Linear
        proj = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        proj = self.batchnorm(proj)
        # LeakyReLU
        direction = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dot = (proj * direction).sum(2, keepdims=True)
        mask = (dot >= 0).float()
        dir_norm_sq = (direction * direction).sum(2, keepdims=True)
        x_out = self.negative_slope * proj + (1 - self.negative_slope) * (
            mask * proj + (1 - mask) * (proj - (dot / (dir_norm_sq + EPS)) * direction)
        )
        return x_out


class VNBatchNorm(nn.Module):
    """Vector Neuron batch normalization."""

    def __init__(self, num_features: int, dim: int):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point features of shape [B, N_feat, 3, N_samples, ...]
        """
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x


def mean_pool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Mean pooling over specified dimension."""
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    """Vector Neuron standard frame feature extractor.

    Extracts SO(3)-invariant features by projecting onto a learned frame.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 4,
        normalize_frame: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        frame_dim: int = 3,
    ):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)

        if frame_dim > 3:
            self.vn_lin = nn.Linear(in_channels // 4, frame_dim, bias=False)
        else:
            if normalize_frame:
                self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
            else:
                self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Point features of shape [B, N_feat, 3, N_samples, ...]

        Returns:
            Tuple of (standardized features, frame vectors)
        """
        feat = x
        feat = self.vn1(feat)
        feat = self.vn2(feat)
        feat = self.vn_lin(feat.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # Gram-Schmidt orthogonalization
            v1 = feat[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)

            v2 = feat[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)

            # Third vector via cross product
            u3 = torch.cross(u1, u2)
            feat = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            feat = feat.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, feat)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, feat)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, feat)

        return x_std, feat
