"""Loss functions for RIST training."""

import torch

from external.chamfer import ChamferDistance
from external.emd import EMDDistance


chamfer_fn = ChamferDistance()
emd_fn = EMDDistance()


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Chamfer Distance loss.

    Args:
        pred: Predicted point cloud of shape (B, N, 3)
        target: Target point cloud of shape (B, N, 3)

    Returns:
        Chamfer distance loss
    """
    pred = pred.float()
    target = target.float()

    dist1, dist2, _, _ = chamfer_fn(pred, target)
    # Use squared distance for numerical stability (avoid gradient explosion from sqrt near zero)
    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss


def emd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance loss.

    Args:
        pred: Predicted point cloud of shape (B, N, 3)
        target: Target point cloud of shape (B, N, 3)

    Returns:
        EMD loss
    """
    dist, _ = emd_fn(pred, target, 0.005, 50)
    # Use squared distance for numerical stability (avoid gradient explosion from sqrt near zero)
    loss = dist.mean(1).mean()
    return loss
