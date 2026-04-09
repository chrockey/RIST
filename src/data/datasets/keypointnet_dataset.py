"""KeypointNet dataset implementation."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from src.utils.common import NAMES2ID
from src.data.transforms import PointCloudTransform
from src.data.utils.download import download_splits


def load_pcd(path: Path) -> np.ndarray:
    """Load point cloud from PCD file (ASCII format).

    Args:
        path: Path to PCD file

    Returns:
        Point cloud array of shape (N, 3)
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Find DATA line to determine where points start
    data_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            data_idx = i + 1
            break

    # Parse points (x, y, z, rgb) - we only need xyz
    points = []
    for line in lines[data_idx:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.array(points, dtype=np.float32)


# Constants
MAX_KEYPOINTS = 25
INVALID_KP_IDX = -1


class KeypointNetDataset(Dataset):
    """KeypointNet dataset for keypoint correspondence learning.

    The KeypointNet dataset contains 3D point clouds with annotated keypoints
    for learning semantic correspondence.

    Directory structure expected:
        keypointnet_dir/
            annotations/
                <category>.json  # keypoint annotations for all models in category
            pcds/
                <category_id>/
                    <model_id>.pcd  # point clouds (ASCII PCD format)
    """

    def __init__(
        self,
        data_dir: Path,
        category: str,
        split: str = "train",
        rotate: bool = False,
        repeat: int = 1,
        flip: bool = False,
        flip_p: float = 0.5,
        jitter: bool = False,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
        elastic: bool = False,
        elastic_params: Optional[List[List[float]]] = None,
        elastic_p: float = 0.95,
    ):
        """Initialize KeypointNet dataset.

        Args:
            data_dir: Path to KeypointNet dataset root
            category: Category name (e.g., 'airplane', 'chair')
            split: Dataset split ('train', 'val', 'test')
            rotate: Whether to apply random SO(3) rotation
            repeat: Number of times to repeat the dataset
            flip: Whether to apply random flip along x/y axes
            flip_p: Probability of flipping each axis
            jitter: Whether to apply random jitter
            jitter_sigma: Standard deviation of jitter noise
            jitter_clip: Maximum absolute value of jitter
            elastic: Whether to apply elastic distortion
            elastic_params: List of [granularity, magnitude] pairs for elastic distortion
            elastic_p: Probability of applying elastic distortion
        """
        self.data_dir = Path(data_dir)
        self.category = category
        self.category_id = NAMES2ID.get(category, category)
        self.split = split
        self.rotate = rotate
        self.repeat = repeat

        self.transform = PointCloudTransform(
            normalize=True,
            rotate=rotate,
            flip=flip,
            flip_p=flip_p,
            jitter=jitter,
            jitter_sigma=jitter_sigma,
            jitter_clip=jitter_clip,
            elastic=elastic,
            elastic_params=elastic_params,
            elastic_p=elastic_p,
        )

        # Load split file
        self.samples = self._load_split()

        # Load category annotations for symmetry lookup (used in evaluation)
        self.category_annotations = self._load_category_annotations()

        # For val/test, pre-generate all pairs for consistent evaluation
        if self.split in ("val", "test"):
            assert repeat == 1, "repeat must be 1 for val/test splits"
            self.pairs = self._generate_pairs()
        else:
            self.pairs = None

    def _load_split(self) -> List[Dict[str, Any]]:
        """Load dataset split information."""
        split_file = self.data_dir / "splits" / self.category_id / f"{self.split}.txt"

        # Download splits if not present
        if not split_file.exists():
            download_splits(self.data_dir, self.category_id)

        with open(split_file, "r") as f:
            model_ids = [line.strip() for line in f.readlines() if line.strip()]

        # Load misaligned model IDs (knife category has x-axis flipped models)
        self.misaligned_ids = self._load_misaligned_model_ids()

        # Build samples list
        samples = []
        for model_id in model_ids:
            pcd_path = self.data_dir / "pcds" / self.category_id / f"{model_id}.pcd"
            if pcd_path.exists():
                sample = {
                    "model_id": model_id,
                    "pcd_path": str(pcd_path),
                }
                samples.append(sample)

        return samples

    def _load_misaligned_model_ids(self) -> set:
        """Load model IDs that have alignment issues.

        Some models in KeypointNet have x-axis flipped in the original ShapeNet.
        These are listed in files like 'knife_misaligned.txt'.
        When loading these models, we flip the x-axis to correct the alignment.
        """
        misaligned = set()

        # Check for category-specific misaligned file
        misaligned_file = self.data_dir / f"{self.category}_misaligned.txt"
        if misaligned_file.exists():
            with open(misaligned_file, "r") as f:
                for line in f:
                    model_id = line.strip()
                    if model_id:
                        misaligned.add(model_id)

        return misaligned

    def _load_category_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Load all annotations for the category and build model_id -> annotation map.

        This enables looking up symmetry information for rotation-symmetric keypoints.

        Returns:
            Dictionary mapping model_id to full annotation dict (including symmetries)
        """
        annot_file = self.data_dir / "annotations" / f"{self.category}.json"
        if not annot_file.exists():
            return {}

        with open(annot_file, "r") as f:
            annotations = json.load(f)

        return {annot["model_id"]: annot for annot in annotations}

    def _generate_pairs(self) -> List[tuple]:
        """Generate all unique pairs for evaluation.

        Returns:
            List of (src_idx, tgt_idx) tuples, N*(N-1)/2 pairs total
        """
        pairs = []
        N = len(self.samples)
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
        return pairs

    def _load_keypoint_indices(self, model_id: str) -> np.ndarray:
        """Load keypoint point indices for a model from category annotations.

        Args:
            model_id: Model ID to lookup

        Returns:
            Array of point indices of shape (MAX_KEYPOINTS,), -1 for invalid
        """
        annot = self.category_annotations.get(model_id, {})
        keypoints = annot.get("keypoints", [])

        result = np.full(MAX_KEYPOINTS, INVALID_KP_IDX, dtype=np.int64)
        for i, kp in enumerate(keypoints[:MAX_KEYPOINTS]):
            if isinstance(kp, dict) and "pcd_info" in kp:
                result[i] = kp["pcd_info"].get("point_index", INVALID_KP_IDX)

        return result

    def __len__(self) -> int:
        if self.pairs is not None:
            return len(self.pairs)
        return len(self.samples) * self.repeat

    def _load_points(self, pcd_path: str) -> np.ndarray:
        """Load point cloud from file (supports both .pcd and .npy formats).

        Args:
            pcd_path: Path to point cloud file

        Returns:
            Point cloud array of shape (N, 3)
        """
        path = Path(pcd_path)
        if path.suffix == ".pcd":
            return load_pcd(path)
        else:
            points = np.load(pcd_path)
            if points.shape[1] != 3:
                points = points[:, :3]
            return points.astype(np.float32)

    def _load_sample(self, sample_idx: int) -> tuple:
        """Load and transform a single sample.

        Returns:
            Tuple of (points, kp_indices)
        """
        sample = self.samples[sample_idx]
        model_id = sample["model_id"]

        points = self._load_points(sample["pcd_path"])
        if model_id in self.misaligned_ids:
            points[:, 0] = -points[:, 0]
        points = self.transform(points)

        kp_indices = self._load_keypoint_indices(model_id)
        return points, kp_indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.

        For training: returns a single transformed point cloud
        For evaluation: returns a pair of point clouds with keypoint indices
        """
        # For evaluation, use pre-generated pairs
        if self.pairs is not None:
            src_idx, tgt_idx = self.pairs[idx]

            src_points, src_kp_indices = self._load_sample(src_idx)
            tgt_points, tgt_kp_indices = self._load_sample(tgt_idx)

            return {
                "src.pcd": torch.from_numpy(src_points),
                "src.kp_indices": torch.from_numpy(src_kp_indices),
                "tgt.pcd": torch.from_numpy(tgt_points),
                "tgt.kp_indices": torch.from_numpy(tgt_kp_indices),
            }

        # For training, return single sample
        points, _ = self._load_sample(idx % len(self.samples))
        return {
            "pcd": torch.from_numpy(points),
        }
