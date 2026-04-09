"""Point cloud transforms for data augmentation."""

import random
from typing import List, Optional

import numpy as np
import scipy.interpolate
import scipy.ndimage
from scipy.spatial.transform import Rotation


class RandomFlip:
    """Randomly flip point cloud along x axis (+y is up, -z is front)."""

    def __init__(self, p: float = 0.5):
        """Initialize RandomFlip.

        Args:
            p: Probability of flipping x axis
        """
        self.p = p

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply random flip to point cloud.

        Args:
            points: Point cloud of shape (N, 3)

        Returns:
            Flipped point cloud of shape (N, 3)
        """
        if np.random.rand() < self.p:
            points[:, 0] = -points[:, 0]  # Flip x only (preserve front direction)
        return points


class RandomJitter:
    """Add random Gaussian noise to point coordinates."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        """Initialize RandomJitter.

        Args:
            sigma: Standard deviation of Gaussian noise
            clip: Maximum absolute value of noise
        """
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply jitter to point cloud.

        Args:
            points: Point cloud of shape (N, 3)

        Returns:
            Jittered point cloud of shape (N, 3)
        """
        jitter = np.clip(
            self.sigma * np.random.randn(points.shape[0], 3),
            -self.clip,
            self.clip,
        )
        points = points + jitter.astype(points.dtype)
        return points


class ElasticDistortion:
    """Apply elastic distortion to point cloud."""

    def __init__(
        self,
        distortion_params: Optional[List[List[float]]] = None,
        p: float = 0.95,
    ):
        """Initialize ElasticDistortion.

        Args:
            distortion_params: List of [granularity, magnitude] pairs.
                Default: [[0.2, 0.4], [0.8, 1.6]]
            p: Probability of applying distortion
        """
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )
        self.p = p

    @staticmethod
    def elastic_distortion(
        coords: np.ndarray, granularity: float, magnitude: float
    ) -> np.ndarray:
        """Apply elastic distortion on sparse coordinate space.

        Args:
            coords: Point coordinates of shape (N, 3)
            granularity: Size of the noise grid (in same scale as the voxel grid)
            magnitude: Noise multiplier

        Returns:
            Distorted coordinates of shape (N, 3)
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing
        for _ in range(2):
            noise = scipy.ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimension
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords = coords + interp(coords) * magnitude
        return coords

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply elastic distortion to point cloud.

        Args:
            points: Point cloud of shape (N, 3)

        Returns:
            Distorted point cloud of shape (N, 3)
        """
        if self.distortion_params is not None and random.random() < self.p:
            for granularity, magnitude in self.distortion_params:
                points = self.elastic_distortion(points, granularity, magnitude)
        return points


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere centered at origin.

    Args:
        points: Point cloud of shape (N, 3)

    Returns:
        Normalized point cloud
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / max_dist
    return points


class PointCloudTransform:
    """Composable point cloud transform."""

    def __init__(
        self,
        normalize: bool = True,
        rotate: bool = False,
        flip: bool = False,
        flip_p: float = 0.5,
        jitter: bool = False,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
        elastic: bool = False,
        elastic_params: Optional[List[List[float]]] = None,
        elastic_p: float = 0.95,
    ):
        """Initialize PointCloudTransform.

        Args:
            normalize: Whether to normalize to unit sphere
            rotate: Whether to apply random SO(3) rotation
            flip: Whether to apply random flip along x/y axes
            flip_p: Probability of flipping each axis
            jitter: Whether to apply random jitter
            jitter_sigma: Standard deviation of jitter noise
            jitter_clip: Maximum absolute value of jitter
            elastic: Whether to apply elastic distortion
            elastic_params: List of [granularity, magnitude] pairs for elastic distortion
            elastic_p: Probability of applying elastic distortion
        """
        self.normalize = normalize
        self.rotate = rotate

        # Initialize augmentation transforms
        self.flip_transform = RandomFlip(p=flip_p) if flip else None
        self.jitter_transform = RandomJitter(sigma=jitter_sigma, clip=jitter_clip) if jitter else None
        self.elastic_transform = ElasticDistortion(distortion_params=elastic_params, p=elastic_p) if elastic else None

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply transforms to point cloud.

        Args:
            points: Point cloud of shape (N, 3)

        Returns:
            Transformed points of shape (N, 3)
        """
        # Normalize
        if self.normalize:
            points = normalize_point_cloud(points)

        # Apply rotation
        if self.rotate:
            R = Rotation.random().as_matrix().astype(np.float32)
            points = points @ R.T

        # Apply augmentations
        if self.flip_transform is not None:
            points = self.flip_transform(points)

        if self.jitter_transform is not None:
            points = self.jitter_transform(points)

        if self.elastic_transform is not None:
            points = self.elastic_transform(points)

        return points.astype(np.float32)
