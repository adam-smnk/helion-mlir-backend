"""MLIR lowering module for device IR operations."""

from __future__ import annotations

from .control_flow import build_kernel_body
from .control_flow import lower_nested_for_loop
from .host_tensor_ops import lower_host_tensor
from .host_tensor_ops import materialize_host_tensor_alias_shape
from .host_tensor_ops import resolve_host_tensor_alias_value
from .load_ops import lower_flat_gather
from .load_slice_ops import lower_load
from .matmul_ops import emit_matmul_like
from .matmul_ops import lower_baddbmm
from .matmul_ops import lower_matmul
from .memory_ops import lower_getitem
from .memory_ops import lower_store
from .subscript_ops import lower_subscript
from .tensor_creation_ops import lower_full
from .tensor_creation_ops import lower_zeros
from .tile_index_ops import lower_tile_index

__all__ = [
    "build_kernel_body",
    "emit_matmul_like",
    "lower_baddbmm",
    "lower_flat_gather",
    "lower_full",
    "lower_getitem",
    "lower_host_tensor",
    "lower_load",
    "lower_matmul",
    "lower_nested_for_loop",
    "lower_store",
    "lower_subscript",
    "lower_tile_index",
    "lower_zeros",
    "materialize_host_tensor_alias_shape",
    "resolve_host_tensor_alias_value",
]
