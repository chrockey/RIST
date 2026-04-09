"""KeypointNet DataModule for PyTorch Lightning."""

import logging
from pathlib import Path
from typing import List, Optional

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from lightning import LightningDataModule

from src.data.datasets.keypointnet_dataset import KeypointNetDataset
from src.utils.common import NAMES2ID

logger = logging.getLogger(__name__)


class KeypointNetDataModule(LightningDataModule):
    """LightningDataModule for KeypointNet dataset.

    Handles data loading for keypoint correspondence learning task.
    """

    def __init__(
        self,
        keypointnet_dir: str = "./data/KeypointNet",
        category: str = "airplane",
        rotate: bool = False,
        batch_size: int = 16,
        num_workers: int = 8,
        pin_memory: bool = True,
        auto_download: bool = True,
        force_download: bool = False,
        name: str = "keypointnet",
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
        """Initialize KeypointNet DataModule.

        Args:
            keypointnet_dir: Path to KeypointNet dataset
            category: Category name (e.g., 'airplane', 'chair')
            rotate: Whether to apply random SO(3) rotation
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            auto_download: Whether to auto-download dataset if missing
            force_download: If True, delete existing data and re-download
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
        super().__init__()
        self.save_hyperparameters()

        self.keypointnet_dir = Path(keypointnet_dir)
        self.category = category
        self.category_id = NAMES2ID.get(category, category)
        self.rotate = rotate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.auto_download = auto_download
        self.force_download = force_download
        self.repeat = repeat
        self.flip = flip
        self.flip_p = flip_p
        self.jitter = jitter
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.elastic = elastic
        self.elastic_params = elastic_params
        self.elastic_p = elastic_p

        self.ds_train: Optional[KeypointNetDataset] = None
        self.ds_val: Optional[KeypointNetDataset] = None
        self.ds_test: Optional[KeypointNetDataset] = None

    def prepare_data(self) -> None:
        """Download dataset if needed and verify structure."""
        # Extract any manually downloaded zip files first
        self._extract_zips()

        if not self._verify_dataset():
            if self.auto_download:
                self._download_dataset()
            else:
                raise FileNotFoundError(
                    f"KeypointNet dataset not found at {self.keypointnet_dir}. "
                    f"Set auto_download=True to download automatically, or "
                    f"download manually from: "
                    f"https://drive.google.com/drive/folders/1_d1TzZEF25Wy5kRj5ZugrgGeyf7xxu8F"
                )

        if not self._verify_dataset():
            raise RuntimeError(
                f"KeypointNet dataset structure is invalid. "
                f"Expected: annotations/{self.category}.json, "
                f"pcds/{self.category_id}/"
            )

    def _extract_zips(self) -> None:
        """Extract any zip files in the dataset directory.

        This handles the case where users manually downloaded zip files
        but did not extract them.
        """
        if not self.keypointnet_dir.exists():
            return

        from src.data.utils.download import extract_zip

        # Known zip files and their expected extraction directories
        zip_configs = [
            ("pcds.zip", "pcds"),
            ("ShapeNetCore.v2.ply.zip", "ShapeNetCore.v2.ply"),
        ]

        for zip_name, marker_dir_name in zip_configs:
            zip_path = self.keypointnet_dir / zip_name
            marker_dir = self.keypointnet_dir / marker_dir_name

            if zip_path.exists() and not marker_dir.exists():
                logger.info(f"Extracting {zip_path}...")
                print(f"Extracting {zip_name}... (this may take a while)")
                try:
                    extract_zip(zip_path, self.keypointnet_dir)
                    zip_path.unlink()
                    logger.info(f"Extracted and removed {zip_name}")
                    print(f"Extraction complete: {zip_name}")
                except Exception as e:
                    logger.error(f"Failed to extract {zip_path}: {e}")
                    raise RuntimeError(
                        f"Failed to extract {zip_path}. "
                        f"Please manually extract it or delete and re-download."
                    ) from e

    def _verify_dataset(self) -> bool:
        """Verify dataset exists with proper structure."""
        if not self.keypointnet_dir.exists():
            return False

        required_paths = [
            self.keypointnet_dir / "annotations" / f"{self.category}.json",
            self.keypointnet_dir / "pcds" / self.category_id,
        ]
        return all(p.exists() for p in required_paths)

    def _download_dataset(self) -> None:
        """Download the KeypointNet dataset."""
        from src.data.utils.download import download_keypointnet

        download_keypointnet(
            output_dir=self.keypointnet_dir,
            category=self.category,
            category_id=self.category_id,
            quiet=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.ds_train = KeypointNetDataset(
                data_dir=self.keypointnet_dir,
                category=self.category,
                split="train",
                rotate=self.rotate,
                repeat=self.repeat,
                flip=self.flip,
                flip_p=self.flip_p,
                jitter=self.jitter,
                jitter_sigma=self.jitter_sigma,
                jitter_clip=self.jitter_clip,
                elastic=self.elastic,
                elastic_params=self.elastic_params,
                elastic_p=self.elastic_p,
            )
            self.ds_val = KeypointNetDataset(
                data_dir=self.keypointnet_dir,
                category=self.category,
                split="val",
                rotate=self.rotate,
            )

        if stage == "test" or stage is None:
            self.ds_test = KeypointNetDataset(
                data_dir=self.keypointnet_dir,
                category=self.category,
                split="test",
                rotate=self.rotate,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        sampler = None
        shuffle = True

        if dist.is_initialized():
            sampler = DistributedSampler(self.ds_train, shuffle=True)
            shuffle = False

        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
                    )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
                    )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
                    )

    def __repr__(self) -> str:
        return f"KeypointNetDataModule(category={self.category}, rotate={self.rotate})"
