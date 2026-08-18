"""Error handling and diagnostics for MLIR backend.

Provides custom exceptions and diagnostic utilities for better error messages
and debugging support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx


class MLIRBackendError(Exception):
    """Base exception for all MLIR backend errors."""


class UnsupportedOperationError(MLIRBackendError):
    """Raised when an unsupported operation is encountered."""

    def __init__(
        self,
        op_name: str,
        reason: str | None = None,
        alternatives: list[str] | None = None,
    ) -> None:
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


class ShapeError(MLIRBackendError):
    """Raised when shape inference fails."""

    def __init__(
        self,
        shape: object,
        reason: str | None = None,
        constraint: str | None = None,
    ) -> None:
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

    def __init__(self, shape: object, symbol_name: str | None = None) -> None:
        reason = f"Dynamic shape {symbol_name or 'with SymInt'} not yet supported"
        constraint = "Use static_shapes=True in @helion.kernel decorator"
        super().__init__(shape, reason, constraint)
        self.symbol_name = symbol_name


class ValueNotFoundError(MLIRBackendError):
    """Raised when a value lookup fails."""

    def __init__(self, node: object, context: str | None = None) -> None:
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
    ) -> None:
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
    ) -> None:
        msg = f"Module building failed at stage: {stage}"
        if reason:
            msg += f"\nReason: {reason}"
        if recovery_hint:
            msg += f"\nHint: {recovery_hint}"
        super().__init__(msg)
        self.stage = stage
        self.reason = reason
        self.recovery_hint = recovery_hint


def safe_int_conversion(val: object, param_name: str = "value") -> int:
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
