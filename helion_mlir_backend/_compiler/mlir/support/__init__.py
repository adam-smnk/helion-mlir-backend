"""Support utilities for MLIR code generation.

This module contains utility functions and classes used throughout the MLIR
compilation pipeline:
- block_ids: Block dimension identification and mapping
- type_utils: Type conversion between torch and MLIR
- errors: Error types and diagnostics
- node_dispatch: Helion-specific node lowering dispatch
- aten_prepass: ATen operation preprocessing and helper generation
"""

from __future__ import annotations

from .aten_prepass import refresh_aten_tensor_meta
from .block_ids import SCALAR_SYMBOL_KINDS
from .block_ids import block_id_from_key
from .block_ids import symbol_origin_info
from .errors import DynamicShapeError
from .errors import ModuleBuilderError
from .errors import NodeLoweringError
from .errors import ShapeError
from .errors import UnsupportedOperationError
from .errors import ValueNotFoundError
from .errors import safe_int_conversion
from .index_meta import IndexDescriptor
from .index_meta import resolve_index_descriptor
from .node_dispatch import lower_helion_node
from .symbolic_shape_restoration import restore_symbolic_shapes_in_bodies
from .type_utils import get_zero_attr
from .type_utils import mlir_dtype_to_torch
from .type_utils import torch_dtype_to_mlir
from .type_utils import torch_tensor_to_mlir_type

__all__ = [
    "SCALAR_SYMBOL_KINDS",
    "DynamicShapeError",
    "IndexDescriptor",
    "ModuleBuilderError",
    "NodeLoweringError",
    "ShapeError",
    "UnsupportedOperationError",
    "ValueNotFoundError",
    "block_id_from_key",
    "get_zero_attr",
    "lower_helion_node",
    "mlir_dtype_to_torch",
    "refresh_aten_tensor_meta",
    "resolve_index_descriptor",
    "restore_symbolic_shapes_in_bodies",
    "safe_int_conversion",
    "symbol_origin_info",
    "torch_dtype_to_mlir",
    "torch_tensor_to_mlir_type",
]
