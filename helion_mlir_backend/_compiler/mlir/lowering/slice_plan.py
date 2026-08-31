"""Authoritative per-dimension slice plan from Helion index metadata.

Instead of re-deriving geometry from sizes/extents or heuristics, a SlicePlan
captures what each index position actually means (block id, scalar, full slice)
and emits one canonical (offsets, static_sizes, strides) tuple for tensor
extract/insert operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


@dataclass(frozen=True)
class DimSlice:
    """Descriptor of a single source/destination dimension slice."""

    kind: Literal["scalar", "tile", "full"]
    offset: ir.Value
    size: int
    block_id: int | None = None
    reduces: bool = False


@dataclass(frozen=True)
class SlicePlan:
    """Complete descriptor of a load/store operation's index→dimension mapping."""

    dims: list[DimSlice]

    def offsets(self) -> list:
        """Dynamic offset values for extract/insert_slice, one per base-tensor dimension."""
        return [dim.offset for dim in self.dims]

    def static_sizes(self) -> list[int]:
        """Static extent sizes for extract/insert_slice, one per base-tensor dimension."""
        return [dim.size for dim in self.dims]

    def value_shape(self) -> list[int]:
        """Shape of the loaded/stored value tile (reduced dims omitted)."""
        return [dim.size for dim in self.dims if not dim.reduces]

    def reduced_dims(self) -> list[int]:
        """Dimension indices that are scalar-indexed and dropped from the result."""
        return [i for i, dim in enumerate(self.dims) if dim.reduces]


def plan_slice(
    ctx: BuildContext,
    index_nodes: list | tuple,
    base_type: ir.RankedTensorType,
) -> SlicePlan:
    """Build a SlicePlan from authoritative index metadata.

    Every index position d maps to base tensor dimension d. For each index:
    - Full slice (None) → full: offset 0, size = base extent.
    - Scalar index (grid/tile.begin) → scalar: block_id from symbol, size 1, reduces.
    - Tile index (block_id) → tile: block_id from symbol, size = block size.
    - Literal int → scalar constant offset, size 1, reduces.

    Raises NodeLoweringError if a tile index cannot be resolved to a block id.
    """
    import mlir.ir as ir

    from ..support.errors import NodeLoweringError

    base_rank = len(base_type.shape)
    sym_to_block_id = ctx.build_sym_to_block_id()
    dims: list[DimSlice] = []

    for dimension, index_node in enumerate(index_nodes):
        if dimension >= base_rank:
            break

        # Full slice: contribute the full base dimension.
        if isinstance(index_node, slice):
            dims.append(
                DimSlice(
                    kind="full",
                    offset=ctx.index_const(0),
                    size=int(base_type.shape[dimension]),
                    block_id=None,
                    reduces=False,
                )
            )
            continue

        # Scalar index: grid/tile.begin or literal int.
        if ctx.is_scalar_index_node(index_node):
            block_id: int | None = None
            scalar_value: ir.Value | None = None

            # Resolve from symbol origin if available.
            symbol_info = ctx.node_symbol_info(index_node)
            if symbol_info is not None:
                block_id = symbol_info[0]

            # Fallback: try to get the value directly.
            if block_id is None:
                scalar_value = ctx.get_value(index_node)
            elif block_id in ctx.block_id_to_iv:
                scalar_value = ctx.block_id_to_iv[block_id]

            offset = (
                ctx.cast_to_index(scalar_value)
                if scalar_value is not None
                else ctx.index_const(0)
            )
            # If the base tensor's own extent here is 1, this dimension has
            # already been reduced to a single local slot (e.g. a synthetic
            # per-iteration accumulator); any offset other than 0 would be
            # out of bounds, regardless of the index's absolute block id/iv.
            if int(base_type.shape[dimension]) == 1:
                offset = ctx.index_const(0)
            dims.append(
                DimSlice(
                    kind="scalar",
                    offset=offset,
                    size=1,
                    block_id=block_id,
                    reduces=True,
                )
            )
            continue

        # Tile index: must resolve to a block id.
        block_id, bias = ctx.infer_index_block_and_bias(index_node, sym_to_block_id)
        if block_id is None:
            raise NodeLoweringError(
                index_node,
                reason=f"Tile index at dimension {dimension} has no resolvable block id",
                recovery_hint="Ensure all tile indices are in hl.tile() loops with configured block_sizes",
            )

        # Compute the tile offset and size.
        if block_id in ctx.block_id_to_iv:
            offset = ctx.block_id_to_iv[block_id]
            if bias:
                offset = ir.ops.arith.addi(offset, ctx.index_const(bias))
        else:
            offset = ctx.index_const(bias)

        tile_size = ctx.block_id_to_size.get(block_id, 1)
        upper_bound = ctx.block_id_to_upper_bound.get(block_id)
        base_extent = int(base_type.shape[dimension])

        if upper_bound is not None:
            tile_size = min(tile_size, upper_bound)
        tile_size = min(tile_size, base_extent)

        # Same local-slot invariant as the scalar case above: if the base
        # tensor's extent here exactly equals one tile's worth, this
        # dimension has already been reduced to a single local tile by an
        # enclosing loop (e.g. a synthetic per-iteration accumulator), so the
        # offset into it must be 0 rather than the absolute block iv/bias.
        if base_extent == tile_size:
            offset = ctx.index_const(0)

        dims.append(
            DimSlice(
                kind="tile",
                offset=offset,
                size=tile_size,
                block_id=block_id,
                reduces=False,
            )
        )

    # Pad to base rank if needed (remaining dims are full slices).
    while len(dims) < base_rank:
        dims.append(
            DimSlice(
                kind="full",
                offset=ctx.index_const(0),
                size=int(base_type.shape[len(dims)]),
                block_id=None,
                reduces=False,
            )
        )

    return SlicePlan(dims)
