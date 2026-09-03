"""MLIR lowering module for device IR operations."""

from __future__ import annotations

from .control_flow import build_kernel_body
from .control_flow import lower_nested_for_loop
from .einsum_ops import lower_einsum
from .host_tensor_ops import lower_host_tensor
from .host_tensor_ops import materialize_host_tensor_alias_shape
from .host_tensor_ops import resolve_host_tensor_alias_value
from .load_ops import lower_flat_gather
from .load_slice_ops import lower_load
from .matmul_ops import emit_matmul_like
from .matmul_ops import lower_baddbmm
from .matmul_ops import lower_matmul
from .matmul_ops import resolve_contraction_operand
from .memory_ops import lower_getitem
from .memory_ops import lower_store
from .subscript_ops import lower_subscript
from .tensor_creation_ops import lower_full
from .tensor_creation_ops import lower_zeros
from .tile_index_ops import lower_tile_index
from .tile_index_ops import lower_tile_scalar_op
from .tile_index_ops import scalar_tile_value
from .transpose_ops import is_transpose_node
from .transpose_ops import lower_transpose
from .transpose_ops import swaps_last_two_dims

__all__ = [
    "build_kernel_body",
    "emit_matmul_like",
    "is_transpose_node",
    "lower_baddbmm",
    "lower_einsum",
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
    "lower_tile_scalar_op",
    "lower_transpose",
    "lower_zeros",
    "materialize_host_tensor_alias_shape",
    "resolve_contraction_operand",
    "resolve_host_tensor_alias_value",
    "scalar_tile_value",
    "swaps_last_two_dims",
]
