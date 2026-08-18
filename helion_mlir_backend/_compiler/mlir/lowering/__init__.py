"""MLIR lowering module for device IR operations."""

from __future__ import annotations

from .control_flow import build_kernel_body
from .control_flow import lower_nested_for_loop
from .load_ops import lower_flat_gather
from .load_slice_ops import lower_load
from .matmul_ops import emit_matmul_like
from .matmul_ops import lower_baddbmm
from .matmul_ops import lower_matmul
from .memory_ops import lower_getitem
from .memory_ops import lower_store
from .subscript_ops import lower_subscript

__all__ = [
    "build_kernel_body",
    "emit_matmul_like",
    "lower_baddbmm",
    "lower_flat_gather",
    "lower_getitem",
    "lower_load",
    "lower_matmul",
    "lower_nested_for_loop",
    "lower_store",
    "lower_subscript",
]
