"""Lower Helion tile-index operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..support.errors import UnsupportedOperationError
from ..support.type_utils import torch_dtype_to_mlir

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
    symbols = ctx.build_sym_to_block_id()
    block_id = ctx.infer_block_id_from_index(tile_argument, symbols)
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
