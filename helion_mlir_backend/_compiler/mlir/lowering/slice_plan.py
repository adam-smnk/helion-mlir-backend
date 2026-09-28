"""Authoritative per-dimension slice plan from Helion index metadata.

Instead of re-deriving geometry from sizes/extents or heuristics, a SlicePlan
captures what each index position actually means (block id, scalar, full slice)
and gives one canonical (offsets, sizes) pair for tensor extract/insert
operations. A tile has a static extent (see ``BlockGeometry.tile_extent``); at the
end of its loop or tensor only its first ``size`` elements are real (loads
zero-pad the rest, stores drop it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from . import emit

if TYPE_CHECKING:
    from ..build_context import BuildContext
    from .tensor_state import OwnedDim


@dataclass(frozen=True)
class DimSlice:
    """Descriptor of a single source/destination dimension slice."""

    kind: Literal["scalar", "tile", "full", "gather"]
    offset: ir.Value
    size: emit.Size
    """Elements taken from the tensor; fewer than ``tile`` at a boundary."""
    tile: int
    """Static extent of this dimension in the loaded or stored tile."""
    block_id: int | None = None
    reduces: bool = False
    index: ir.Value | None = None
    """For ``gather``: the index tensor (the slice spans the whole dimension)."""


@dataclass(frozen=True)
class SlicePlan:
    """Complete descriptor of a load/store operation's index→dimension mapping."""

    dims: list[DimSlice]

    def offsets(self) -> list:
        """Dynamic offset values for extract/insert_slice, one per base-tensor dimension."""
        return [dim.offset for dim in self.dims]

    def sizes(self) -> list[emit.Size]:
        """Slice sizes for extract/insert_slice, one per base-tensor dimension."""
        return [dim.size for dim in self.dims]

    def tile_shape(self) -> list[int]:
        """Static shape of the tile at full rank."""
        return [dim.tile for dim in self.dims]

    def value_shape(self) -> list[int]:
        """Shape of the loaded/stored value tile (reduced dims omitted)."""
        return [dim.tile for dim in self.dims if not dim.reduces]

    def value_sizes(self) -> list[emit.Size]:
        """Real part of the value tile (reduced dims omitted)."""
        return [dim.size for dim in self.dims if not dim.reduces]

    def is_partial(self) -> bool:
        """Whether the tile may extend past its loop or tensor."""
        return any(
            isinstance(dim.size, ir.Value) or dim.size != dim.tile for dim in self.dims
        )

    def reduced_dims(self) -> list[int]:
        """Dimension indices that are scalar-indexed and dropped from the result."""
        return [i for i, dim in enumerate(self.dims) if dim.reduces]

    def gathers(self) -> list[tuple[int, ir.Value]]:
        """``(dimension, index tensor)`` of each tensor-indexed dimension."""
        return [
            (i, dim.index) for i, dim in enumerate(self.dims) if dim.kind == "gather"
        ]


def plan_slice(
    ctx: BuildContext,
    index_nodes: list | tuple,
    base: ir.Value,
    owned: dict[int, OwnedDim] | None = None,
) -> SlicePlan:
    """Build a SlicePlan from authoritative index metadata.

    Every index position d maps to base tensor dimension d. For each index:
    - Full slice (None) → full: offset 0, size = base extent.
    - Scalar index (grid/tile.begin) → scalar: block_id from symbol, size 1, reduces.
    - Tile index (block_id) → tile: block_id from symbol, tile-extent wide.
    - Literal int → scalar constant offset, size 1, reduces.
    - Index tensor → gather: the whole dimension, gathered after slicing.

    ``owned`` marks dims where ``base`` is only the current iteration's region
    of a larger tensor (see ``tensor_state``); those are indexed from its origin.

    Raises NodeLoweringError if a tile index cannot be resolved to a block id.
    """
    from ..support.errors import NodeLoweringError
    from ..support.index_meta import resolve_index_descriptor

    owned = owned or {}
    base_shape = ir.RankedTensorType(base.type).shape
    base_rank = len(base_shape)
    dims: list[DimSlice] = []

    def full(start: int, size: int) -> DimSlice:
        return DimSlice("full", ctx.index_const(start), size, size)

    for dimension, index_node in enumerate(index_nodes):
        if dimension >= base_rank:
            break
        extent = base_shape[dimension]

        if isinstance(index_node, slice):
            start, stop = _static_slice_bounds(index_node, extent)
            dims.append(full(start, stop - start))
            continue

        descriptor = resolve_index_descriptor(ctx, index_node)
        owner = owned.get(dimension)
        if owner is not None:
            if descriptor.block_id != owner.block_id or descriptor.bias:
                raise NodeLoweringError(
                    index_node,
                    reason=(
                        f"dimension {dimension} is owned by block_id "
                        f"{owner.block_id} but indexed by another expression"
                    ),
                )
            zero = ctx.index_const(0)
            if owner.point:
                dims.append(DimSlice("scalar", zero, 1, 1, owner.block_id, True))
                continue
            # The region is the tile's real part; its extent is dynamic when partial.
            size = (
                tensor_d.DimOp(base, ctx.index_const(dimension)).result
                if ir.ShapedType.is_dynamic_size(extent)
                else extent
            )
            tile = ctx.geometry.tile_extent(owner.block_id)
            dims.append(DimSlice("tile", zero, size, tile, owner.block_id))
            continue

        if descriptor.is_scalar:
            block_id = descriptor.block_id
            if descriptor.is_offset and block_id in ctx.block_id_to_iv:
                scalar_value = ctx.block_id_to_iv[block_id]
            else:
                scalar_value = ctx.get_value(index_node)
            offset = (
                ctx.cast_to_index(scalar_value)
                if scalar_value is not None
                else ctx.index_const(descriptor.bias)
            )
            dims.append(DimSlice("scalar", offset, 1, 1, block_id, reduces=True))
            continue

        block_id, bias = descriptor.block_id, descriptor.bias
        index = ctx.get_value(index_node) if block_id is None else None
        if index is not None and isinstance(index.type, ir.RankedTensorType):
            dims.append(
                DimSlice("gather", ctx.index_const(0), extent, extent, index=index)
            )
            continue
        if block_id is None:
            raise NodeLoweringError(
                index_node,
                reason=f"Tile index at dimension {dimension} has no resolvable block id",
                recovery_hint="Ensure all tile indices are in hl.tile() loops with configured block_sizes",
            )
        if block_id not in ctx.geometry.blocks:
            raise NodeLoweringError(
                index_node,
                reason=f"Tile index at dimension {dimension} names unknown block_id {block_id}",
            )
        offset, size, tile = tile_window(ctx, block_id, bias, extent)
        dims.append(DimSlice("tile", offset, size, tile, block_id))

    while len(dims) < base_rank:
        dims.append(full(0, base_shape[len(dims)]))

    return SlicePlan(dims)


def tile_window(
    ctx: BuildContext, block_id: int, bias: int, extent: int
) -> tuple[ir.Value, emit.Size, int]:
    """Offset, real size and tile size of ``block_id``'s current tile, shifted by
    ``bias``, in a dimension of ``extent``.

    The real part is what lies inside both the loop and the tensor. The tensor
    bound is only checked when the loop may run past it (e.g. an input smaller
    than the iteration domain, whose excess is then read as zeros).
    """
    from ..support.errors import UnsupportedOperationError

    tile = ctx.geometry.tile_extent(block_id)
    offset = ctx.block_id_to_iv.get(block_id)
    if offset is None:
        raise UnsupportedOperationError(
            "tile index", reason=f"block_id {block_id} is used outside its loop"
        )
    if bias:
        offset = ir.ops.arith.addi(offset, ctx.index_const(bias))
    size = ctx.block_id_to_valid.get(block_id, tile)
    begin, end = ctx.block_id_to_bounds[block_id]
    if bias < 0 and not (isinstance(begin, int) and begin + bias >= 0):
        raise UnsupportedOperationError(
            "tile index", reason=f"offset {bias} may index before the tensor start"
        )
    if isinstance(end, int) and end + bias <= extent:
        return offset, size, tile
    d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
    offset = emit.affine_min([d0, ir.AffineConstantExpr.get(extent)], [offset])
    remaining = ir.AffineConstantExpr.get(extent) - d0
    if isinstance(size, int):
        size = emit.affine_min([ir.AffineConstantExpr.get(size), remaining], [offset])
    else:
        size = emit.affine_min([remaining, d1], [offset, size])
    return offset, size, tile


def _static_slice_bounds(index: slice, extent: int) -> tuple[int, int]:
    """``[start, stop)`` of a unit-step slice with static bounds, Python-style."""
    from ..support.errors import UnsupportedOperationError

    bounds = []
    for value in (index.start, index.stop):
        try:
            bounds.append(None if value is None else int(value))
        except (TypeError, ValueError):
            bounds.append(value)
    if index.step not in (None, 1) or not all(
        bound is None or isinstance(bound, int) for bound in bounds
    ):
        raise UnsupportedOperationError(
            "subscript slice", reason=f"only static unit-step slices, got {index}"
        )
    start, stop, _ = slice(*bounds).indices(extent)
    return start, max(start, stop)
