# Earth Mover's Distance (EMD) CUDA Extension

This module provides a fast CUDA implementation of Earth Mover's Distance for 3D point clouds.

## Installation

```bash
cd src/external/emd
python setup.py install
```

## Requirements

- CUDA Toolkit
- PyTorch with CUDA support

## Usage

```python
from src.external.emd.emd_module import emdModule

emd = emdModule()
dist, assignment = emd(points1, points2, eps=0.005, iters=50)
```

## Fallback

If the CUDA extension is not compiled, Chamfer Distance will be used as a fallback.
