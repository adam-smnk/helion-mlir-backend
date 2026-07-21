"""Dynamic shape support and SymInt handling for MLIR backend.

Provides utilities for working with dynamic shapes and SymInt values in the
MLIR lowering pipeline. This enables better handling of symbolic dimensions
while still generating valid MLIR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    import mlir.ir as ir

log = logging.getLogger(__name__)


class SymbolInfo:
    """Information about a symbolic dimension."""

    def __init__(self, name: str, symbol: Any, block_id: Optional[int] = None):
        """Initialize symbol info.

        Parameters
        ----------
        name : str
            Symbolic name (e.g., 'u0', 'm', 'n')
        symbol : Any
            The actual SymInt or torch.SymInt object
        block_id : Optional[int]
            Associated block_id if this is a tiled dimension
        """
        self.name = name
        self.symbol = symbol
        self.block_id = block_id
        self.concrete_value: Optional[int] = None

    def try_resolve(self) -> Optional[int]:
        """Try to resolve the symbol to a concrete value.

        Returns
        -------
        Optional[int]
            Concrete value if resolvable, None otherwise
        """
        if self.concrete_value is not None:
            return self.concrete_value

        try:
            # Attempt direct conversion
            if isinstance(self.symbol, int):
                self.concrete_value = self.symbol
                return self.symbol

            # Try torch.SymInt conversion
            if isinstance(self.symbol, torch.SymInt):
                self.concrete_value = int(self.symbol)
                return self.concrete_value

            # Try hasattr __index__
            if hasattr(self.symbol, "__index__"):
                self.concrete_value = self.symbol.__index__()
                return self.concrete_value
        except Exception as e:
            log.debug("Could not resolve symbol %s: %s", self.name, e)

        return None

    def __repr__(self) -> str:
        return f"SymbolInfo(name={self.name}, block_id={self.block_id}, concrete={self.concrete_value})"


class SymbolTable:
    """Table of symbolic dimensions encountered during lowering."""

    def __init__(self):
        """Initialize symbol table."""
        self._symbols: Dict[str, SymbolInfo] = {}
        self._symbol_to_concrete: Dict[str, int] = {}

    def register_symbol(
        self,
        name: str,
        symbol: Any,
        block_id: Optional[int] = None,
    ) -> SymbolInfo:
        """Register a new symbol.

        Parameters
        ----------
        name : str
            Name of the symbol
        symbol : Any
            The symbol object (SymInt, int, etc.)
        block_id : Optional[int]
            Associated block_id if tiled

        Returns
        -------
        SymbolInfo
            Information about the registered symbol
        """
        if name in self._symbols:
            return self._symbols[name]

        info = SymbolInfo(name, symbol, block_id)
        self._symbols[name] = info

        # Try to resolve immediately
        concrete = info.try_resolve()
        if concrete is not None:
            self._symbol_to_concrete[name] = concrete
            log.debug("Resolved symbol %s to %d", name, concrete)

        return info

    def get_symbol(self, name: str) -> Optional[SymbolInfo]:
        """Get symbol info by name."""
        return self._symbols.get(name)

    def get_concrete_value(self, name: str, default: int = 0) -> int:
        """Get concrete value for a symbol.

        Parameters
        ----------
        name : str
            Name of the symbol
        default : int
            Default value if symbol cannot be resolved

        Returns
        -------
        int
            Concrete value or default
        """
        if name in self._symbol_to_concrete:
            return self._symbol_to_concrete[name]

        info = self.get_symbol(name)
        if info is not None:
            concrete = info.try_resolve()
            if concrete is not None:
                self._symbol_to_concrete[name] = concrete
                return concrete

        return default

    def all_symbols(self) -> List[SymbolInfo]:
        """Get all registered symbols."""
        return list(self._symbols.values())

    def __repr__(self) -> str:
        return f"SymbolTable({len(self._symbols)} symbols)"


def extract_symbol_from_shape(
    shape_node: Any,
    index: int,
    symbol_table: SymbolTable,
) -> Tuple[Optional[str], Optional[int]]:
    """Extract symbol name and try to resolve shape dimension.

    Parameters
    ----------
    shape_node : Any
        FX node or value from shape tuple
    index : int
        Position in shape tuple
    symbol_table : SymbolTable
        Symbol table for registration

    Returns
    -------
    Tuple[Optional[str], Optional[int]]
        (symbol_name, concrete_value) or (None, default)
    """
    if isinstance(shape_node, torch.fx.Node):
        # Check if this node produces a SymInt
        meta_val = shape_node.meta.get("val")
        if isinstance(meta_val, torch.SymInt):
            sym_name = str(meta_val)
            info = symbol_table.register_symbol(sym_name, meta_val)
            concrete = info.try_resolve()
            return sym_name, concrete if concrete is not None else 1

    # Direct value
    if isinstance(shape_node, torch.SymInt):
        sym_name = str(shape_node)
        info = symbol_table.register_symbol(sym_name, shape_node)
        concrete = info.try_resolve()
        return sym_name, concrete if concrete is not None else 1

    if isinstance(shape_node, int):
        return None, shape_node

    return None, 1  # Default


def create_dynamic_shape_indexing(
    shape: List[int],
    symbol_names: Dict[int, Optional[str]],
) -> ir.Attribute:
    """Create MLIR indexing for a shape with dynamic dimensions.

    Parameters
    ----------
    shape : List[int]
        List of concrete dimension values (1 if unknown)
    symbol_names : Dict[int, Optional[str]]
        Mapping from shape index to symbol name (if dynamic)

    Returns
    -------
    ir.Attribute
        MLIR shape attribute
    """
    # For Linalg-on-Tensors, we need concrete shapes.
    # If dimension is symbolic (symbol_names[i] is not None),
    # we use the concrete fallback value or '?' for unknown dimensions.
    return shape


def generate_shape_dependent_code(
    shape_dims: List[Tuple[Optional[str], int]],
    callback,
) -> Any:
    """Generate code that handles both static and dynamic shape scenarios.

    Parameters
    ----------
    shape_dims : List[Tuple[Optional[str], int]]
        List of (symbol_name, concrete_value) for each dimension
    callback : callable
        Callback to generate code for given shape

    Returns
    -------
    Any
        Result from callback
    """
    # Extract concrete shape for current code path
    concrete_shape = [concrete for _, concrete in shape_dims]

    # Generate code for this concrete shape
    result = callback(concrete_shape)

    # Log which dimensions were dynamic
    dynamic_dims = [i for i, (sym_name, _) in enumerate(shape_dims) if sym_name is not None]
    if dynamic_dims:
        log.debug("Generated code with dynamic dimensions: %s", dynamic_dims)

    return result


def validate_shape_compatibility(
    shape1: List[Tuple[Optional[str], int]],
    shape2: List[Tuple[Optional[str], int]],
) -> bool:
    """Check if two shapes are compatible (same symbolic/concrete structure).

    Parameters
    ----------
    shape1 : List[Tuple[Optional[str], int]]
        First shape
    shape2 : List[Tuple[Optional[str], int]]
        Second shape

    Returns
    -------
    bool
        True if compatible
    """
    if len(shape1) != len(shape2):
        return False

    for (sym1, concrete1), (sym2, concrete2) in zip(shape1, shape2):
        # If both are symbolic with same name, OK
        if sym1 is not None and sym2 is not None and sym1 == sym2:
            continue
        # If both are concrete and equal, OK
        if sym1 is None and sym2 is None and concrete1 == concrete2:
            continue
        # Otherwise mismatch
        return False

    return True


def create_symbolic_shape_annotation(
    shape_dims: List[Tuple[Optional[str], int]],
) -> str:
    """Create a human-readable annotation for a shape.

    Parameters
    ----------
    shape_dims : List[Tuple[Optional[str], int]]
        List of (symbol_name, concrete_value)

    Returns
    -------
    str
        Shape annotation string
    """
    parts = []
    for sym_name, concrete in shape_dims:
        if sym_name is not None:
            parts.append(f"{sym_name}({concrete})")
        else:
            parts.append(str(concrete))
    return f"[{', '.join(parts)}]"
