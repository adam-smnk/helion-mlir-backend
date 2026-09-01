"""Lower Helion tile-index operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..support import UnsupportedOperationError
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

    tile_argument = node.args[0]
    block_id = ctx.infer_block_id_from_index(tile_argument)
    if block_id is None and ctx.for_block_id_stack:
        block_id = ctx.for_block_id_stack[-1]

    shape = ctx.shape_from_node_meta(node) or []
    if not shape:
        if block_id is None or block_id not in ctx.block_id_to_size:
            return None
        shape = [ctx.block_id_to_size[block_id]]

    if block_id is not None and block_id in ctx.block_id_to_size:
        shape[0] = int(ctx.block_id_to_size[block_id])
    if (
        block_id is not None
        and block_id in ctx.block_id_to_upper_bound
        and ctx.block_id_to_upper_bound[block_id] > 0
    ):
        shape[0] = min(shape[0], int(ctx.block_id_to_upper_bound[block_id]))

    if len(shape) != 1:
        raise UnsupportedOperationError(
            "tile_index",
            reason="Only 1D tile.index lowering is implemented",
        )

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

    if block_id is not None and block_id in ctx.block_id_to_iv:
        base = ctx.block_id_to_iv[block_id]
    else:
        base = ctx.index_const(0)

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
    """Build the scalar ``index`` value for a grid or tile position op."""
    from mlir.dialects import arith as arith_d

    size = ctx.block_id_to_size.get(block_id)
    upper_bound = ctx.block_id_to_upper_bound.get(block_id)

    if kind == "block_size":
        return None if size is None else ctx.index_const(int(size))

    if kind == "tile_count":
        if not size or size <= 0 or not upper_bound or upper_bound <= 0:
            return None
        return ctx.index_const(-(-int(upper_bound) // int(size)))

    induction_variable = ctx.block_id_to_iv.get(block_id)
    if induction_variable is None:
        return None

    if kind in ("grid", "tile_begin"):
        return induction_variable

    if kind == "tile_end":
        if not size or size <= 0:
            return None
        end = arith_d.AddIOp(induction_variable, ctx.index_const(int(size))).result
        # Only the last tile can overrun, and only when the extent is known.
        if upper_bound and upper_bound > 0 and int(upper_bound) % int(size):
            end = arith_d.MinSIOp(end, ctx.index_const(int(upper_bound))).result
        return end

    if kind == "tile_id":
        if not size or size <= 0:
            return None
        if int(size) == 1:
            return induction_variable
        return arith_d.DivUIOp(induction_variable, ctx.index_const(int(size))).result

    return None


def lower_tile_scalar_op(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``tile.begin`` / ``tile.end`` / ``tile.id`` / ``tile.count``."""
    from ..support import block_id_from_key

    kind = getattr(node.target, "__name__", "")
    info = ctx.node_symbol_info(node)
    block_id = info[0] if info is not None else None

    if block_id is None and node.args:
        tile_argument = node.args[0]
        if isinstance(tile_argument, torch.fx.Node) and tile_argument.args:
            block_id = block_id_from_key(tile_argument.args[0])
        if block_id is None:
            block_id = ctx.infer_block_id_from_index(tile_argument)
    if block_id is None and ctx.for_block_id_stack:
        block_id = ctx.for_block_id_stack[-1]
    if block_id is None:
        return None

    return scalar_tile_value(ctx, block_id, kind)
