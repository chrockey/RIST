"""RIST Lightning Module for training SO(3)-invariant correspondence.

Architecture (paper notation):
    P (input point cloud)
        -> Encoder -> z (SO(3)-equivariant global shape descriptor),
                      theta (SO(3)-invariant local shape transform params)
        -> v = z @ theta (SO(3)-equivariant local shape descriptors)
        -> Decoder(v) -> P' (reconstructed point cloud)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only

from torchmetrics import MaxMetric, MeanMetric

from src.losses import chamfer_loss, emd_loss
from src.data.datasets.keypointnet_dataset import INVALID_KP_IDX
from src.utils.metrics import PCKCurve
from external.knn import knn_query


class RISTModule(LightningModule):
    """RIST: Rotation-Invariant Local Shape Transform."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        scheduler_interval: str = "epoch",
        sr_epochs: int = 10,
        lambda_mse: float = 1000.0,
        lambda_emd: float = 1.0,
        lambda_cd: float = 10.0,
    ):
        """Initialize RIST module.

        Args:
            encoder: Pre-configured encoder module.
            decoder: Pre-configured decoder module.
            optimizer: Partial optimizer (functools.partial).
            scheduler: Partial scheduler (functools.partial). If None, no scheduler is used.
            scheduler_interval: Scheduler step interval ('epoch' or 'step').
            sr_epochs: Number of epochs for self-reconstruction only warmup phase.
            lambda_mse: Weight for MSE loss in self-reconstruction.
            lambda_emd: Weight for EMD loss in self-reconstruction.
            lambda_cd: Weight for Chamfer distance loss in cross-reconstruction.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"], logger=False)

        self.mse_loss = nn.MSELoss()
        self.encoder = encoder
        self.decoder = decoder

        # Metrics for validation and test
        self.val_pck = PCKCurve()
        self.test_pck = PCKCurve()
        self.best_auc = MaxMetric()
        self.val_by_uv_dist = MeanMetric()

    def on_fit_start(self) -> None:
        """Validate hyperparameters before training."""
        max_epochs = self.trainer.max_epochs
        if self.hparams.sr_epochs >= max_epochs:
            raise ValueError(
                f"sr_epochs ({self.hparams.sr_epochs}) must be "
                f"less than max_epochs ({max_epochs}) for cross-reconstruction to run."
            )

    def forward(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud to latent representations.

        Args:
            p: Input point cloud (B, N, 3)

        Returns:
            z: SO(3)-equivariant global shape descriptor
            theta: SO(3)-invariant local shape transform parameters
        """
        return self.encoder(p)

    def compute_loss(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Loss formulation (paper notation):
            L_SR = λ_MSE * MSE(P, P') + λ_EMD * EMD(P, P')
            L_CR = λ_CD * CD(P1, P'_2→1)
            L_total = L_SR + L_CR

        Args:
            batch: Batch dictionary containing 'points'.
            epoch: Current epoch index.

        Returns:
            Dictionary with loss_sr, loss_cr, and loss_total.
        """
        p = batch["pcd"]  # (B, N, 3)

        # Encode: P -> (z, theta)
        z, theta = self.encoder(p)

        # Self-reconstruction: v = z @ theta -> P'
        p_recon = self.decoder(z, theta)
        loss_sr = (
            self.hparams.lambda_mse * self.mse_loss(p_recon, p)
            + self.hparams.lambda_emd * emd_loss(p_recon, p)
        )

        # Cross-reconstruction: v_cross = z_i @ theta_j -> P'_cross (after warmup)
        # Use all unique pairs (i < j) instead of circular shift
        loss_cr = torch.tensor(0.0, dtype=p.dtype, device=p.device)
        if epoch >= self.hparams.sr_epochs:
            B = p.shape[0]
            idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=p.device)

            z_i = z[idx_i]
            theta_j = theta[idx_j]
            p_i = p[idx_i]

            p_cross = self.decoder(z_i, theta_j)
            loss_cr = self.hparams.lambda_cd * chamfer_loss(p_cross, p_i)

        return {
            "loss_sr": loss_sr,
            "loss_cr": loss_cr,
            "loss_total": loss_sr + loss_cr,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        losses = self.compute_loss(batch, self.current_epoch)

        self.log_dict({
            "train/loss_total": losses["loss_total"],
            "train/loss_sr": losses["loss_sr"],
            "train/loss_cr": losses["loss_cr"],
        })

        return losses["loss_total"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step with KeyPointNet evaluation."""
        if "src.kp_indices" in batch:
            distances = self.eval_keypointnet(batch)
            self.val_pck.update(distances)
            self.val_by_uv_dist.update(distances)

    def on_validation_epoch_end(self) -> None:
        """Log PCK AUC and curve at end of validation epoch."""
        pck_metrics = self.val_pck.compute()
        auc = pck_metrics["pck_auc"]
        self.log("val/auc", auc)

        by_uv_dist = self.val_by_uv_dist.compute()
        self.log("val/by_uv_dist", by_uv_dist)

        self.best_auc.update(auc)
        self.val_pck.reset()
        self.val_by_uv_dist.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step with KeyPointNet evaluation."""
        if "src.kp_indices" in batch:
            distances = self.eval_keypointnet(batch)
            self.test_pck.update(distances)

    def on_test_epoch_end(self) -> None:
        """Log PCK AUC and curve at end of test epoch."""
        pck_metrics = self.test_pck.compute()
        self.log("test/auc", pck_metrics["pck_auc"])
        self._log_pck_curve("test", self.test_pck)
        self.test_pck.reset()

    @rank_zero_only
    def _log_pck_curve(self, prefix: str, pck_metric: PCKCurve) -> None:
        """Log PCK curve as wandb line plot (rank 0 only).

        Args:
            prefix: Metric prefix (e.g., "val" or "test").
            pck_metric: PCKCurve metric instance.
        """
        if not (self.logger and hasattr(self.logger, "experiment")):
            return
        import wandb

        thresholds, pck_values = pck_metric.compute_curve()
        table = wandb.Table(data=[[t, p] for t, p in zip(thresholds, pck_values)], columns=["threshold", "pck"])
        self.logger.experiment.log({
            f"{prefix}/pck_curve": wandb.plot.line(table, "threshold", "pck")
        })

    @torch.inference_mode()
    def eval_keypointnet(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate keypoint correspondence on KeypointNet."""
        src, tgt = batch["src.pcd"], batch["tgt.pcd"]
        src_kp_idx = batch["src.kp_indices"]
        tgt_kp_idx = batch["tgt.kp_indices"]
        B, N, _ = src.shape
        K = src_kp_idx.shape[1]

        src_kp = torch.gather(src, 1, src_kp_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3))
        tgt_kp = torch.gather(tgt, 1, tgt_kp_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3))

        # Cross-reconstruct: swap theta (shape) between src and tgt
        z, theta = self.encoder(torch.cat([src, tgt], dim=0))
        theta_cross = torch.roll(theta, shifts=B, dims=0)
        p_cross = self.decoder(z, theta_cross)
        src_cross, tgt_cross = p_cross[:B], p_cross[B:]  # src_cross: src pose + tgt shape

        src_flat = src.reshape(-1, 3).contiguous()
        tgt_flat = tgt.reshape(-1, 3).contiguous()
        src_cross_flat = src_cross.reshape(-1, 3).contiguous()
        tgt_cross_flat = tgt_cross.reshape(-1, 3).contiguous()
        src_kp_flat = src_kp.reshape(-1, 3).contiguous()
        tgt_kp_flat = tgt_kp.reshape(-1, 3).contiguous()

        offset_p = torch.arange(1, B + 1, device=src.device) * N
        offset_kp = torch.arange(1, B + 1, device=src.device) * K

        # Find correspondences: tgt_cross has src shape, src_cross has tgt shape
        # knn_query returns global indices into the flat point cloud (0 ~ B*N-1)
        nn_idx_src, _ = knn_query(1, tgt_cross_flat, offset_p, tgt_kp_flat, offset_kp)
        nn_idx_tgt, _ = knn_query(1, src_cross_flat, offset_p, src_kp_flat, offset_kp)
        nn_idx_src = nn_idx_src.squeeze(-1).long()
        nn_idx_tgt = nn_idx_tgt.squeeze(-1).long()

        src_kp_pred = src_flat[nn_idx_src].view(B, K, 3)
        tgt_kp_pred = tgt_flat[nn_idx_tgt].view(B, K, 3)

        # Compute distance between predicted and ground truth keypoints
        dist_src = torch.linalg.norm(src_kp - src_kp_pred, dim=-1)
        dist_tgt = torch.linalg.norm(tgt_kp - tgt_kp_pred, dim=-1)

        # Normalize by diameter
        diam_src = torch.linalg.norm(src.max(1).values - src.min(1).values, dim=-1, keepdim=True)
        diam_tgt = torch.linalg.norm(tgt.max(1).values - tgt.min(1).values, dim=-1, keepdim=True)
        dist_src, dist_tgt = dist_src / diam_src, dist_tgt / diam_tgt

        valid = (src_kp_idx != INVALID_KP_IDX) & (tgt_kp_idx != INVALID_KP_IDX)
        return torch.cat([dist_src[valid], dist_tgt[valid]])

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            # Calculate total_steps for OneCycleLR
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(optimizer=optimizer, total_steps=total_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
