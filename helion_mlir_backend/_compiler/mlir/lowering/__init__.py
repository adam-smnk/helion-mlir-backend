"""MLIR lowering of device IR nodes.

Importing this package registers every handler with :mod:`.registry`.
"""

from __future__ import annotations

from . import contraction_ops as contraction_ops
from . import elementwise_ops as elementwise_ops
from . import view_ops as view_ops
from .control_flow import build_kernel_body
from .control_flow import lower_nested_for_loop
from .host_tensor_ops import lower_host_tensor
from .load_slice_ops import lower_load
from .memory_ops import lower_store
from .registry import lower_node
from .subscript_ops import lower_subscript
from .tensor_creation_ops import lower_full
from .tile_index_ops import lower_tile_index
from .tile_index_ops import scalar_tile_value
from .transpose_ops import lower_transpose

__all__ = [
    "build_kernel_body",
    "lower_full",
    "lower_host_tensor",
    "lower_load",
    "lower_nested_for_loop",
    "lower_node",
    "lower_store",
    "lower_subscript",
    "lower_tile_index",
    "lower_transpose",
    "scalar_tile_value",
]
