"""Direct MLIR lowering for Helion's matmul-family ATen operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def emit_matmul_like(
    ctx: BuildContext,
    lhs: ir.Value,
    rhs: ir.Value,
    out: ir.Value | None = None,
) -> ir.Value | None:
    """Emit rank-2 matmul or rank-3 batch-matmul with an optional output."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    lhs_type = ir.RankedTensorType(lhs.type)
    rhs_type = ir.RankedTensorType(rhs.type)
    if lhs_type.element_type != rhs_type.element_type:
        return None

    lhs_rank = len(lhs_type.shape)
    rhs_rank = len(rhs_type.shape)
    if lhs_rank == 2 and rhs_rank == 2:
        if lhs_type.shape[1] != rhs_type.shape[0]:
            return None
        out_shape = [int(lhs_type.shape[0]), int(rhs_type.shape[1])]
        out_rank = 2
        op_name = "matmul"
    elif lhs_rank == 3 and rhs_rank == 3:
        if (
            lhs_type.shape[0] != rhs_type.shape[0]
            or lhs_type.shape[2] != rhs_type.shape[1]
        ):
            return None
        out_shape = [
            int(lhs_type.shape[0]),
            int(lhs_type.shape[1]),
            int(rhs_type.shape[2]),
        ]
        out_rank = 3
        op_name = "batch_matmul"
    else:
        return None

    if out is None:
        out = tensor_d.EmptyOp(out_shape, lhs_type.element_type).result
        elem_type = lhs_type.element_type
        if isinstance(elem_type, ir.FloatType):
            zero_attr = ir.FloatAttr.get(elem_type, 0.0)
        elif isinstance(elem_type, ir.IntegerType):
            zero_attr = ir.IntegerAttr.get(elem_type, 0)
        else:
            return None
        zero = arith_d.ConstantOp(
            elem_type,
            zero_attr,
        ).result
        out = linalg_d.fill(zero, outs=[out])
    else:
        out_type = ir.RankedTensorType(out.type)
        if out_type.element_type != lhs_type.element_type:
            return None
        if len(out_type.shape) != out_rank or list(out_type.shape) != out_shape:
            return None

    if op_name == "matmul":
        return linalg_d.matmul(lhs, rhs, outs=[out])
    return linalg_d.batch_matmul(lhs, rhs, outs=[out])


def lower_matmul(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``aten.mm``, ``aten.matmul``, or ``aten.bmm``."""
    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if len(args) < 2:
        return None
    lhs = ctx.get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
    rhs = ctx.get_value(args[1]) if isinstance(args[1], torch.fx.Node) else None
    if lhs is None or rhs is None:
        return None
    return emit_matmul_like(ctx, lhs, rhs)


def lower_baddbmm(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``aten.baddbmm`` when its scale factors are one."""
    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if len(args) < 3:
        return None
    acc = ctx.get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
    lhs = ctx.get_value(args[1]) if isinstance(args[1], torch.fx.Node) else None
    rhs = ctx.get_value(args[2]) if isinstance(args[2], torch.fx.Node) else None
    if acc is None or lhs is None or rhs is None:
        return None
    beta = args[3] if len(args) > 3 else 1
    alpha = args[4] if len(args) > 4 else 1
    if beta != 1 or alpha != 1:
        return None
    return emit_matmul_like(ctx, lhs, rhs, out=acc)
