"""Matrix-vector product (gemv) on Helion's MLIR backend.

Implemented as a matmul against a zero-padded ``B`` rather than a dedicated
1-D-output reduction kernel: Helion's own gemv-shaped kernel (a single
``hl.tile`` output dim reducing a full row via ``torch.einsum``) crashes for
bf16 at LLVM translation -- the register-tiling schedule materializes the
2-D reduction operand as an ``!llvm.array<N x vector<Mxbf16>>`` stack buffer
and casts it to a native ``vector<NxMxbf16>`` for ``vector.contract``, and
that specific array-to-vector cast has no LLVM lowering pattern for bf16.
Padding ``B`` out to the AMX block width and reusing the proven
:func:`matmul` kernel sidesteps that broken codegen path entirely and keeps
real (AMX) vectorization instead of forcing the scalar (non-AMX) pipeline,
which works but is ~7x slower than this at M=2048,K=8192 bf16 (measured).
The tradeoff is BLOCK_N-times more FLOPs than a true gemv; still faster than
the scalar-pipeline workaround, though slower than eager PyTorch's
memory-bound gemv for this shape.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .matmul import BLOCK_N
from .matmul import matmul


def supports_matvec(a: Tensor, b: Tensor) -> bool:
    """Whether the padded-matmul gemv path can handle this pair of operands."""
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.bfloat16):
        return False
    if a.device.type != "cpu" or b.device.type != "cpu":
        return False
    if a.dim() != 2 or b.dim() != 2 or b.shape[1] != 1:
        return False
    return a.shape[1] == b.shape[0]


def matvec(a: Tensor, b: Tensor) -> Tensor:
    """``A @ b`` for ``A: [M, K]``, ``b: [K, 1]`` -> ``[M, 1]``.

    Raises ``ValueError`` if unsupported (see :func:`supports_matvec`) instead
    of silently falling back to eager PyTorch -- callers that need a fallback
    must check :func:`supports_matvec` themselves and choose one explicitly.
    """
    if not supports_matvec(a, b):
        raise ValueError(
            f"matvec() does not support these operands: a.shape={tuple(a.shape)} "
            f"a.dtype={a.dtype} a.device={a.device}, b.shape={tuple(b.shape)} "
            f"b.dtype={b.dtype} b.device={b.device}. b must be K-by-1, dtype must be "
            "float32 or bfloat16, both on cpu. Check supports_matvec() before "
            "calling, or use torch.matmul directly for unsupported cases."
        )
    k = a.shape[1]
    b_padded = torch.zeros(k, BLOCK_N, dtype=b.dtype, device=a.device)
    b_padded[:, 0] = b[:, 0]
    return matmul(a, b_padded)[:, :1]
