"""Lower Helion tile-index operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..support import NodeLoweringError
from ..support import torch_dtype_to_mlir

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_tile_index(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``tile.index`` to a one-dimensional tensor of offsets."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    if not node.args:
        return None

    block_id = ctx.infer_block_id_from_index(node.args[0])
    if block_id is None or block_id not in ctx.geometry.blocks:
        raise NodeLoweringError(node, reason="cannot resolve the tile's block id")
    base = ctx.block_id_to_iv.get(block_id)
    if base is None:
        raise NodeLoweringError(
            node, reason=f"block_id {block_id} has no active loop offset"
        )
    shape = [ctx.geometry.tile_extent(block_id)]

    element_type: ir.Type = ir.IndexType.get()
    metadata_value = node.meta.get("val")
    if isinstance(metadata_value, torch.Tensor):
        try:
            metadata_type = torch_dtype_to_mlir(metadata_value.dtype)
            if isinstance(metadata_type, (ir.IntegerType, ir.IndexType)):
                element_type = metadata_type
        except Exception:
            # Best-effort element-type refinement; fall back to index type.
            pass

    index_type = ir.IndexType.get()
    result_type = ir.RankedTensorType.get(shape, element_type)
    operation = tensor_d.GenerateOp(result_type, [])
    body = operation.operation.regions[0].blocks.append(index_type)

    with ir.InsertionPoint(body):
        induction_variable = body.arguments[0]
        if isinstance(element_type, ir.IndexType):
            value = arith_d.AddIOp(base, induction_variable).result
        else:
            base_int = arith_d.IndexCastOp(element_type, base).result
            induction_int = arith_d.IndexCastOp(element_type, induction_variable).result
            value = arith_d.AddIOp(base_int, induction_int).result
        tensor_d.YieldOp(value)

    return operation.result


def scalar_tile_value(ctx: BuildContext, block_id: int, kind: str) -> ir.Value | None:
    """Build the scalar ``index`` value for a grid or tile position op.

    Matches Helion: ``begin`` is the tile's absolute offset, ``end`` is clamped
    to the loop end, ``id`` is ``begin // block_size`` and ``count`` is
    ``cdiv(end - begin, block_size)``.
    """
    from mlir.dialects import arith as arith_d

    geometry = ctx.geometry
    if block_id not in geometry.blocks:
        return None
    block = geometry.block(block_id)
    block_size = block.block_size
    bounds = ctx.block_id_to_bounds.get(block_id)

    if kind == "block_size":
        return ctx.index_const(block_size)

    if kind == "tile_count":
        if block.span is not None:
            return ctx.index_const(geometry.trip_count(block_id))
        if bounds is None:
            return None
        extent = arith_d.SubIOp(ctx.as_index(bounds[1]), ctx.as_index(bounds[0])).result
        return arith_d.CeilDivUIOp(extent, ctx.index_const(block_size)).result

    offset = ctx.block_id_to_iv.get(block_id)
    if offset is None:
        return None

    if kind in ("grid", "tile_begin"):
        return offset

    if kind == "tile_end":
        end = arith_d.AddIOp(offset, ctx.index_const(block_size)).result
        if block.span is None or geometry.is_ragged(block_id):
            if bounds is None:
                return None
            end = arith_d.MinSIOp(end, ctx.as_index(bounds[1])).result
        return end

    if kind == "tile_id":
        trip = ctx.block_id_to_trip_iv.get(block_id)
        if (
            trip is not None
            and bounds is not None
            and isinstance(bounds[0], int)
            and bounds[0] == 0
        ):
            return trip
        if block_size == 1:
            return offset
        return arith_d.DivUIOp(offset, ctx.index_const(block_size)).result

    return None


def lower_tile_scalar_op(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower ``tile.begin`` / ``tile.end`` / ``tile.id`` / ``tile.count``."""
    from ..support import block_id_from_key

    kind = getattr(node.target, "__name__", "")
    if kind == "tile_block_size":
        kind = "block_size"
    info = ctx.node_symbol_info(node)
    block_id = info[0] if info is not None else None

    if block_id is None and node.args:
        tile_argument = node.args[0]
        if isinstance(tile_argument, torch.fx.Node) and tile_argument.args:
            block_id = block_id_from_key(tile_argument.args[0])
        if block_id is None:
            block_id = ctx.infer_block_id_from_index(tile_argument)
    if block_id is None:
        raise NodeLoweringError(node, reason="cannot resolve the tile's block id")

    value = scalar_tile_value(ctx, block_id, kind)
    if value is None:
        raise NodeLoweringError(
            node, reason=f"no active loop bounds for block_id {block_id} ({kind})"
        )
    return value
