"""Error handling and diagnostics for MLIR backend.

Provides custom exceptions and diagnostic utilities for better error messages
and debugging support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch.fx

log = logging.getLogger(__name__)


class MLIRBackendError(Exception):
    """Base exception for all MLIR backend errors."""


class UnsupportedOperationError(MLIRBackendError):
    """Raised when an unsupported operation is encountered."""

    def __init__(
        self,
        op_name: str,
        reason: str | None = None,
        alternatives: list[str] | None = None,
    ):
        msg = f"Unsupported operation: {op_name}"
        if reason:
            msg += f"\nReason: {reason}"
        if alternatives:
            msg += f"\nAlternatives: {', '.join(alternatives)}"
        msg += "\nNote: Check MLIR_LIMITATIONS.md for supported operations"
        super().__init__(msg)
        self.op_name = op_name
        self.reason = reason
        self.alternatives = alternatives


class TypeConversionError(MLIRBackendError):
    """Raised when a type conversion fails."""

    def __init__(
        self,
        torch_type: Any,
        reason: str | None = None,
        supported_types: list[str] | None = None,
    ):
        msg = f"Cannot convert type: {torch_type}"
        if reason:
            msg += f"\nReason: {reason}"
        if supported_types:
            msg += f"\nSupported types: {', '.join(supported_types)}"
        super().__init__(msg)
        self.torch_type = torch_type
        self.reason = reason
        self.supported_types = supported_types


class ShapeError(MLIRBackendError):
    """Raised when shape inference fails."""

    def __init__(
        self,
        shape: Any,
        reason: str | None = None,
        constraint: str | None = None,
    ):
        msg = f"Invalid shape: {shape}"
        if reason:
            msg += f"\nReason: {reason}"
        if constraint:
            msg += f"\nConstraint: {constraint}"
        super().__init__(msg)
        self.shape = shape
        self.reason = reason
        self.constraint = constraint


class DynamicShapeError(ShapeError):
    """Raised when dynamic shapes are encountered but not supported."""

    def __init__(self, shape: Any, symbol_name: str | None = None):
        reason = f"Dynamic shape {symbol_name or 'with SymInt'} not yet supported"
        constraint = "Use static_shapes=True in @helion.kernel decorator"
        super().__init__(shape, reason, constraint)
        self.symbol_name = symbol_name


class ValueNotFoundError(MLIRBackendError):
    """Raised when a value lookup fails."""

    def __init__(self, node: Any, context: str | None = None):
        msg = f"Value not found for node: {node}"
        if context:
            msg += f"\nContext: {context}"
        super().__init__(msg)
        self.node = node
        self.context = context


class NodeLoweringError(MLIRBackendError):
    """Raised when lowering a single FX node fails."""

    def __init__(
        self,
        node: torch.fx.Node,
        reason: str | None = None,
        recovery_hint: str | None = None,
    ):
        msg = f"Failed to lower FX node: {node.op}[{node.name}]"
        if hasattr(node, "target"):
            msg += f"\nTarget: {node.target}"
        if reason:
            msg += f"\nReason: {reason}"
        if recovery_hint:
            msg += f"\nHint: {recovery_hint}"
        super().__init__(msg)
        self.node = node
        self.reason = reason
        self.recovery_hint = recovery_hint


class ModuleBuilderError(MLIRBackendError):
    """Raised when module building fails."""

    def __init__(
        self,
        stage: str,
        reason: str | None = None,
        recovery_hint: str | None = None,
    ):
        msg = f"Module building failed at stage: {stage}"
        if reason:
            msg += f"\nReason: {reason}"
        if recovery_hint:
            msg += f"\nHint: {recovery_hint}"
        super().__init__(msg)
        self.stage = stage
        self.reason = reason
        self.recovery_hint = recovery_hint


class OperandError(MLIRBackendError):
    """Raised when operand types or counts are incorrect."""

    def __init__(
        self,
        operation: str,
        actual: Any,
        expected: Any,
        operand_name: str | None = None,
    ):
        msg = f"Operand mismatch for {operation}"
        if operand_name:
            msg += f" ({operand_name})"
        msg += f"\nExpected: {expected}\nActual: {actual}"
        super().__init__(msg)
        self.operation = operation
        self.actual = actual
        self.expected = expected
        self.operand_name = operand_name


def validate_tensor_shape(shape: Any, allow_dynamic: bool = False) -> None:
    """Validate that a tensor shape is valid.

    Parameters
    ----------
    shape : Any
        Shape tuple/list to validate
    allow_dynamic : bool
        Whether to allow SymInt (dynamic shapes)

    Raises
    ------
    ShapeError
        If shape is invalid
    DynamicShapeError
        If dynamic shapes encountered and not allowed
    """
    if not isinstance(shape, (tuple, list)):
        raise ShapeError(shape, "Shape must be a tuple or list")

    for i, dim in enumerate(shape):
        # Check for SymInt (dynamic shapes)
        if hasattr(dim, "__class__") and "SymInt" in str(dim.__class__):
            if not allow_dynamic:
                raise DynamicShapeError(shape, f"dim[{i}]")
        # Check for valid integers
        elif isinstance(dim, int):
            if dim <= 0:
                raise ShapeError(
                    shape,
                    f"Dimension {i} is non-positive",
                    "All dimensions must be > 0",
                )
        else:
            raise ShapeError(
                shape, f"Dimension {i} is not an integer", f"Got {type(dim)}"
            )


def safe_int_conversion(val: Any, param_name: str = "value") -> int:
    """Safely convert a value to int with helpful error messages.

    Parameters
    ----------
    val : Any
        Value to convert
    param_name : str
        Name of the parameter (for error messages)

    Returns
    -------
    int
        The converted integer value

    Raises
    ------
    TypeError
        If conversion fails
    """
    try:
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            if val != int(val):
                raise ValueError(f"{param_name}={val} is not an integer")
            return int(val)
        # Try torch.SymInt resolution
        if hasattr(val, "__index__"):
            return val.__index__()
        raise TypeError(f"Cannot convert {type(val).__name__} to int")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Failed to convert {param_name} to int: {e}") from e


def diagnose_unsupported_op(op_name: str) -> str:
    """Generate a diagnostic message for an unsupported operation.

    Parameters
    ----------
    op_name : str
        Name of the unsupported operation

    Returns
    -------
    str
        Diagnostic message with suggestions
    """
    # Map of common unsupported ops to alternatives
    unsupported_map = {
        "layer_norm": ("torch.nn.LayerNorm", "Consider fusing with other ops"),
        "softmax": ("torch.nn.Softmax", "Consider fusing into attention pattern"),
        "sigmoid": ("Pointwise ops", "Consider using mul + add for approximation"),
        "exp": ("Pointwise ops", "May require special handling"),
        "log": ("Pointwise ops", "May require special handling"),
        "sort": ("Reduction ops", "Not supported in Linalg-on-Tensors"),
        "scatter": ("Indexing ops", "Consider reshape + gather alternatives"),
        "gather": ("Indexing ops", "Consider reshape + extract_slice alternatives"),
    }

    msg = f"Operation '{op_name}' is not yet supported by the MLIR backend.\n"

    if op_name.lower() in unsupported_map:
        canonical, suggestion = unsupported_map[op_name.lower()]
        msg += f"Canonical name: {canonical}\n"
        msg += f"Suggestion: {suggestion}\n"

    msg += "Supported operations:\n"
    msg += "  - Basic: matmul, addmm, mm, add, mul, relu\n"
    msg += "  - Creation: zeros, full\n"
    msg += "  - Indexing: extract_slice, insert_slice, getitem\n"
    msg += "\nFor more information, see MLIR_LIMITATIONS.md"

    return msg


def log_diagnostic_info(
    stage: str,
    node: torch.fx.Node | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Log diagnostic information for debugging.

    Parameters
    ----------
    stage : str
        Current processing stage
    node : Optional[torch.fx.Node]
        Current FX node being processed
    context : Optional[dict]
        Additional context information
    """
    log.debug(f"Diagnostic: stage={stage}")
    if node:
        log.debug(f"  node.op={node.op}, node.name={node.name}")
        if hasattr(node, "target"):
            log.debug(f"  target={node.target}")
        if hasattr(node, "meta"):
            log.debug(f"  meta_keys={list(node.meta.keys())}")
    if context:
        for key, value in context.items():
            log.debug(f"  {key}={value}")
