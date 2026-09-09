"""Helper utilities for AI-bench Helion MLIR CPU kernels."""

from __future__ import annotations

from .elementwise import leaky_relu
from .elementwise import relu
from .elementwise import scalar_mul
from .linear import CACHE_PREPACKED_WEIGHTS_ENV
from .linear import AffineLinearCache
from .linear import LinearCache
from .linear import linear
from .linear import linear_affine
from .matmul import bmm
from .matmul import identity_epilogue
from .matmul import matmul
from .matmul import matmul_prepacked_b
from .matmul import matmul_prepacked_b_affine
from .matmul import pack_a_blocked
from .matmul import pack_a_blocked_t
from .matmul import pack_b_blocked
from .matmul import pack_b_blocked_t
from .matmul import supports
from .reduction import matvec
from .reduction import supports_matvec

__all__ = [
    "CACHE_PREPACKED_WEIGHTS_ENV",
    "AffineLinearCache",
    "LinearCache",
    "bmm",
    "identity_epilogue",
    "leaky_relu",
    "linear",
    "linear_affine",
    "matmul",
    "matmul_prepacked_b",
    "matmul_prepacked_b_affine",
    "matvec",
    "pack_a_blocked",
    "pack_a_blocked_t",
    "pack_b_blocked",
    "pack_b_blocked_t",
    "relu",
    "scalar_mul",
    "supports",
    "supports_matvec",
]
