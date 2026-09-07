# ruff: noqa: ANN002, ANN003, ANN204, I001, I002, UP008
# Example Helion MLIR CPU kernel
# Status: Experimental / uncurated
# Expectation: Correctness-first, performance not representative
# Signature and style follow the KernelBench reference model verbatim.


import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul(A, B)
