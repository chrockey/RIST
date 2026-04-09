# Chamfer Distance CUDA Extension

This module provides a fast CUDA implementation of Chamfer Distance for 3D point clouds.

## Installation

```bash
cd src/external/chamfer
python setup.py install
```

## Requirements

- CUDA Toolkit
- PyTorch with CUDA support

## Usage

```python
import src.external.chamfer.dist_chamfer_3D as chamfer

cf_dist = chamfer.chamfer_3DDist()
dist1, dist2, idx1, idx2 = cf_dist(points1, points2)
```

## Fallback

If the CUDA extension is not compiled, the pure PyTorch implementation in `src/losses/loss.py` will be used automatically.
