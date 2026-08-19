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
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
) -> ir.Value | None:
    """Emit rank-2 matmul or rank-3 batch-matmul with an optional output.

    Operands may be narrower than the accumulator (bf16/bf16 -> f32 and friends);
    linalg contractions extend them natively.
    """
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
    if lhs_rank != rhs_rank or lhs_rank not in (2, 3):
        return None

    batch = list(lhs_type.shape[:-2])
    if batch != list(rhs_type.shape[:-2]):
        return None

    lhs_rows, lhs_reduction = lhs_type.shape[-2:]
    if transpose_lhs:
        lhs_rows, lhs_reduction = lhs_reduction, lhs_rows
    rhs_reduction, rhs_cols = rhs_type.shape[-2:]
    if transpose_rhs:
        rhs_reduction, rhs_cols = rhs_cols, rhs_reduction
    if lhs_reduction != rhs_reduction:
        return None

    out_shape = [*batch, int(lhs_rows), int(rhs_cols)]
    out_rank = lhs_rank

    if out is None:
        accumulator_type = lhs_type.element_type
        out = tensor_d.EmptyOp(out_shape, accumulator_type).result
        if isinstance(accumulator_type, ir.FloatType):
            zero_attr = ir.FloatAttr.get(accumulator_type, 0.0)
        elif isinstance(accumulator_type, ir.IntegerType):
            zero_attr = ir.IntegerAttr.get(accumulator_type, 0)
        else:
            return None
        zero = arith_d.ConstantOp(
            accumulator_type,
            zero_attr,
        ).result
        out = linalg_d.fill(zero, outs=[out])
    else:
        out_type = ir.RankedTensorType(out.type)
        if not _is_valid_accumulator(lhs_type.element_type, out_type.element_type):
            return None
        if len(out_type.shape) != out_rank or list(out_type.shape) != out_shape:
            return None

    if not transpose_lhs and not transpose_rhs:
        if out_rank == 2:
            return linalg_d.matmul(lhs, rhs, outs=[out])
        return linalg_d.batch_matmul(lhs, rhs, outs=[out])

    return _emit_contract(ctx, lhs, rhs, out, len(batch), transpose_lhs, transpose_rhs)


def _is_valid_accumulator(operand_type: ir.Type, accumulator_type: ir.Type) -> bool:
    """Return whether a contraction may accumulate operands into this type."""
    import mlir.ir as ir

    if operand_type == accumulator_type:
        return True
    if isinstance(operand_type, ir.FloatType) and isinstance(
        accumulator_type, ir.FloatType
    ):
        return accumulator_type.width >= operand_type.width
    if isinstance(operand_type, ir.IntegerType) and isinstance(
        accumulator_type, ir.IntegerType
    ):
        return accumulator_type.width >= operand_type.width
    return False


def _emit_contract(
    ctx: BuildContext,
    lhs: ir.Value,
    rhs: ir.Value,
    out: ir.Value,
    batch_rank: int,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> ir.Value | None:
    """Emit ``linalg.contract`` with indexing maps encoding transposed operands."""
    from mlir.dialects import linalg as linalg_d
    import mlir.ir as ir

    # Iteration space is (batch..., m, n, k).
    total = batch_rank + 3
    batch_dims = [ir.AffineDimExpr.get(i) for i in range(batch_rank)]
    m = ir.AffineDimExpr.get(batch_rank)
    n = ir.AffineDimExpr.get(batch_rank + 1)
    k = ir.AffineDimExpr.get(batch_rank + 2)

    lhs_dims = [k, m] if transpose_lhs else [m, k]
    rhs_dims = [n, k] if transpose_rhs else [k, n]

    maps = [
        ir.AffineMap.get(total, 0, batch_dims + lhs_dims),
        ir.AffineMap.get(total, 0, batch_dims + rhs_dims),
        ir.AffineMap.get(total, 0, [*batch_dims, m, n]),
    ]

    return linalg_d.contract(
        lhs,
        rhs,
        outs=[out],
        indexing_maps=maps,
    )


def resolve_contraction_operand(
    ctx: BuildContext, argument: object
) -> tuple[ir.Value | None, bool]:
    """Resolve a contraction operand, lowering any pending subscript views first."""
    from .subscript_ops import lower_subscript
    from .transpose_ops import is_transpose_node
    from .transpose_ops import swaps_last_two_dims

    if not isinstance(argument, torch.fx.Node):
        return None, False

    value = ctx.get_value(argument)
    if value is None and str(getattr(argument.target, "__name__", "")) == "subscript":
        value = lower_subscript(ctx, argument)
        if value is not None:
            ctx.set_value(argument, value)

    if value is None and argument.args:
        base = argument.args[0]
        if isinstance(base, torch.fx.Node) and ctx.get_value(base) is None:
            lowered = (
                lower_subscript(ctx, base)
                if str(getattr(base.target, "__name__", "")) == "subscript"
                else None
            )
            if lowered is not None:
                ctx.set_value(base, lowered)

    if is_transpose_node(argument) and swaps_last_two_dims(argument):
        base = argument.args[0] if argument.args else None
        base_value = ctx.get_value(base) if isinstance(base, torch.fx.Node) else None
        if base_value is not None:
            return base_value, True

    return value, False


def lower_matmul(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``aten.mm``, ``aten.matmul``, or ``aten.bmm``."""
    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if len(args) < 2:
        return None
    lhs, transpose_lhs = resolve_contraction_operand(ctx, args[0])
    rhs, transpose_rhs = resolve_contraction_operand(ctx, args[1])
    if lhs is None or rhs is None:
        return None
    return emit_matmul_like(
        ctx,
        lhs,
        rhs,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
    )


def lower_baddbmm(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``aten.baddbmm`` when its scale factors are one."""
    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if len(args) < 3:
        return None
    acc = ctx.get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
    lhs, transpose_lhs = resolve_contraction_operand(ctx, args[1])
    rhs, transpose_rhs = resolve_contraction_operand(ctx, args[2])
    if acc is None or lhs is None or rhs is None:
        return None
    beta = args[3] if len(args) > 3 else 1
    alpha = args[4] if len(args) > 4 else 1
    if beta != 1 or alpha != 1:
        return None
    return emit_matmul_like(
        ctx,
        lhs,
        rhs,
        out=acc,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
    )
