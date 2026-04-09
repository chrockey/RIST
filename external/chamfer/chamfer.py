"""Chamfer Distance 3D implementation with JIT compilation."""

import os
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

# JIT compile
print("JIT compiling Chamfer...")
ext_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(ext_dir, "src")
_C = load(
    name="chamfer_C",
    sources=[
        os.path.join(src_dir, "chamfer.cpp"),
        os.path.join(src_dir, "chamfer.cu"),
    ],
    verbose=True,
)
print("Loaded JIT Chamfer CUDA extension")


class ChamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n, device=device)
        dist2 = torch.zeros(batchsize, m, device=device)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int32, device=device)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int32, device=device)

        torch.cuda.set_device(device)
        _C.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size(), device=device)
        gradxyz2 = torch.zeros(xyz2.size(), device=device)

        _C.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return ChamferFunction.apply(input1, input2)


chamfer_distance = ChamferFunction.apply
