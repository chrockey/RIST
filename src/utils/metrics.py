"""Custom metrics for evaluation."""

from typing import List, Optional

import torch
from torchmetrics import Metric


class PCKCurve(Metric):
    """Percentage of Correct Keypoints (PCK) curve metric.

    Computes PCK at multiple thresholds efficiently by accumulating
    hit counts per threshold during training/validation.

    Args:
        thresholds: List of distance thresholds for PCK computation.
                   Default is [0.00, 0.01, ..., 0.20] (21 values).
        dist_sync_on_step: Synchronize metric state across processes at each step.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if thresholds is None:
            thresholds = [i * 0.01 for i in range(21)]  # 0.00, 0.01, ..., 0.20
        self.thresholds = thresholds

        # State: hit counts per threshold and total count
        self.add_state(
            "hits",
            default=torch.zeros(len(thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, distances: torch.Tensor) -> None:
        """Update state with new distances.

        Args:
            distances: Normalized distances of shape (N,) where N is number of keypoints.
                      Should be pre-filtered to exclude invalid keypoints.
        """
        distances = distances.detach()
        for i, t in enumerate(self.thresholds):
            self.hits[i] += (distances < t).sum()
        self.total += distances.numel()

    def compute(self) -> dict:
        """Compute PCK values for all thresholds and AUC.

        Returns:
            Dictionary with PCK@threshold values and PCK_AUC.
        """
        if self.total == 0:
            result = {f"pck@{t:.2f}": 0.0 for t in self.thresholds}
            result["pck_auc"] = 0.0
            return result

        pck_values = (self.hits / self.total * 100).tolist()
        result = {f"pck@{t:.2f}": v for t, v in zip(self.thresholds, pck_values)}

        # Compute AUC using trapezoidal rule (normalized to 0-100 range)
        auc = self._compute_auc(pck_values)
        result["pck_auc"] = auc

        return result

    def _compute_auc(self, pck_values: List[float]) -> float:
        """Compute area under PCK curve using trapezoidal rule.

        Args:
            pck_values: List of PCK percentages at each threshold.

        Returns:
            AUC normalized to 0-100 range.
        """
        if len(self.thresholds) < 2:
            return pck_values[0] if pck_values else 0.0

        # Trapezoidal integration
        auc = 0.0
        for i in range(len(self.thresholds) - 1):
            dx = self.thresholds[i + 1] - self.thresholds[i]
            auc += (pck_values[i] + pck_values[i + 1]) / 2 * dx

        # Normalize by threshold range to get 0-100 scale
        threshold_range = self.thresholds[-1] - self.thresholds[0]
        if threshold_range > 0:
            auc = auc / threshold_range

        return auc

    def compute_curve(self) -> tuple:
        """Compute PCK curve data for plotting.

        Returns:
            Tuple of (thresholds, pck_values) for curve plotting.
        """
        if self.total == 0:
            return self.thresholds, [0.0] * len(self.thresholds)

        pck_values = (self.hits / self.total * 100).tolist()
        return self.thresholds, pck_values
