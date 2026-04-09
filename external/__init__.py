"""External CUDA extensions for RIST.

These modules need to be compiled separately:
- chamfer: Chamfer Distance CUDA extension
- emd: Earth Mover's Distance CUDA extension

To compile:
    cd src/external/chamfer && python setup.py install
    cd src/external/emd && python setup.py install
"""
