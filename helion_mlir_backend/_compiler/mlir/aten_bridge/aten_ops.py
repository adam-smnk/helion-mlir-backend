"""Registry for ATen operations with custom MLIR lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def aten_target_matches(node: torch.fx.Node, *names: str) -> bool:
    """Match ATen targets by canonical name or equivalent runtime FX alias."""
    target_name = str(node.target).lower()
    overload_name = str(getattr(node.target, "__name__", "")).lower()
    target_variants = {
        target_name,
        overload_name,
        target_name.replace("aten.", ""),
        target_name.replace("torch.", ""),
        target_name.replace(".default", ""),
        overload_name.replace(".default", ""),
    }
    for name in names:
        normalized = name.lower()
        if normalized in target_variants:
            return True
        if normalized.replace("aten.", "") in target_variants:
            return True
        if normalized.replace(".default", "") in target_variants:
            return True
        if normalized.replace("aten.", "").replace(".default", "") in {
            s.replace(".default", "") for s in target_variants
        }:
            return True
        base = normalized.replace("aten.", "").replace(".default", "")
        for variant in target_variants:
            if base in variant:
                return True
    return False


def lower_custom_aten(builder: object, node: torch.fx.Node) -> ir.Value | None:
    """Try the registered custom ATen lowerers in precedence order."""
    if aten_target_matches(
        node,
        "aten.view",
        "aten.reshape",
        "view.default",
        "reshape.default",
    ):
        lowered = lower_static_reshape(builder.context, node)
        if lowered is not None:
            return lowered

    if aten_target_matches(node, "aten.addmm", "addmm.default"):
        lowered = lower_addmm(builder.context, node)
        if lowered is not None:
            return lowered

    if aten_target_matches(node, "aten.baddbmm", "baddbmm.default"):
        lowered = builder._lower_aten_baddbmm(node)
        if lowered is not None:
            return lowered

    if aten_target_matches(node, "aten.add.Tensor", "add.Tensor"):
        lowered = lower_add_matmul_accumulate(builder.context, node)
        if lowered is not None:
            return lowered

    lowered = lower_scalar_binary(builder.context, node)
    if lowered is not None:
        return lowered

    if aten_target_matches(
        node,
        "aten.mm",
        "aten.matmul",
        "aten.bmm",
        "mm",
        "mm.default",
        "matmul",
        "matmul.default",
        "bmm",
        "bmm.default",
    ):
        lowered = builder._lower_aten_matmul(node)
        if lowered is not None:
            return lowered

    lowered = lower_transpose_node(builder.context, node)
    if lowered is not None:
        return lowered

    lowered = lower_passthrough(builder.context, node)
    if lowered is not None:
        return lowered

    return None


def lower_static_reshape(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower statically shaped ATen view/reshape without a helper function."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if not args or not isinstance(args[0], torch.fx.Node):
        return None
    source = ctx.get_value(args[0])
    result_shape = ctx.shape_from_node_meta(node)
    if source is None or result_shape is None or not result_shape:
        return None
    if any(dim <= 0 for dim in result_shape):
        return None
    source_type = ir.RankedTensorType(source.type)
    if _shape_product(source_type.shape) != _shape_product(result_shape):
        return None
    result_type = ir.RankedTensorType.get(result_shape, source_type.element_type)
    i32 = ir.IntegerType.get_signless(32)
    shape_type = ir.RankedTensorType.get([len(result_shape)], i32)
    shape_values = [
        arith_d.ConstantOp(i32, ir.IntegerAttr.get(i32, dim)).result
        for dim in result_shape
    ]
    shape = tensor_d.FromElementsOp(shape_type, shape_values).result
    return tensor_d.ReshapeOp(result_type, source, shape).result


def _shape_product(shape: object) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product


def lower_passthrough(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower shape-preserving ATen aliases without emitting a helper call."""
    from ..aten_lowering import normalized_aten_args

    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    if not any(
        operation in target_name
        for operation in ("aten.alias", "aten.detach", "aten.clone", "aten.contiguous")
    ) and overload_name not in {
        "alias.default",
        "detach.default",
        "clone.default",
        "contiguous.default",
    }:
        return None
    args = list(normalized_aten_args(node))
    if args and isinstance(args[0], torch.fx.Node):
        return ctx.get_value(args[0])
    return None


def lower_addmm(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower unit-scaled ``aten.addmm`` into an accumulator matmul."""
    from ..aten_lowering import normalized_aten_args
    from ..lowering import emit_matmul_like
    from ..lowering import resolve_contraction_operand

    args = list(normalized_aten_args(node))
    if len(args) < 3:
        return None
    accumulator = ctx.get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
    lhs, transpose_lhs = resolve_contraction_operand(ctx, args[1])
    rhs, transpose_rhs = resolve_contraction_operand(ctx, args[2])
    if accumulator is None or lhs is None or rhs is None:
        return None
    beta = args[3] if len(args) > 3 else 1
    alpha = args[4] if len(args) > 4 else 1
    if beta != 1 or alpha != 1:
        return None
    lowered = emit_matmul_like(
        ctx,
        lhs,
        rhs,
        out=accumulator,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
    )
    if lowered is not None:
        return lowered
    if transpose_lhs or transpose_rhs:
        return None
    # Shapes recovered from tiled metadata can be imprecise; linalg.matmul still
    # verifies them, so keep emitting it directly rather than falling back.
    from mlir.dialects import linalg as linalg_d

    return linalg_d.matmul(lhs, rhs, outs=[accumulator])


def lower_transpose_node(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower a standalone permute/transpose of a tile."""
    from ..lowering import lower_transpose

    return lower_transpose(ctx, node)


def convert_tensor_element_type(
    ctx: BuildContext, source: ir.Value, target_type: ir.Type
) -> ir.Value | None:
    """Convert a tensor's element type elementwise, returning *source* if equal."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    source_type = ir.RankedTensorType(source.type)
    if str(source_type.element_type) == str(target_type):
        return source

    shape = list(source_type.shape)
    result_type = ir.RankedTensorType.get(shape, target_type)
    generate = tensor_d.GenerateOp(result_type, [])
    body = generate.operation.regions[0].blocks.append(
        *([ir.IndexType.get()] * len(shape))
    )
    with ir.InsertionPoint(body):
        element = tensor_d.ExtractOp(
            source,
            list(body.arguments),
            results=[source_type.element_type],
        ).result
        converted = ctx.cast_scalar_to(element, target_type)
        if converted is None:
            return None
        tensor_d.YieldOp(converted)
    return generate.result


def lower_add_matmul_accumulate(
    ctx: BuildContext, node: torch.fx.Node
) -> ir.Value | None:
    """Lower ``acc + matmul`` into a loop-carried accumulator update."""
    from ..aten_lowering import normalized_aten_args
    from ..lowering import emit_matmul_like

    args = list(normalized_aten_args(node))
    if len(args) < 2 or (args[2] if len(args) > 2 else 1) != 1:
        return None
    accumulator_node: torch.fx.Node | None = None
    matmul_node: torch.fx.Node | None = None
    for first, second in ((args[0], args[1]), (args[1], args[0])):
        if not isinstance(first, torch.fx.Node) or not isinstance(
            second, torch.fx.Node
        ):
            continue
        if aten_target_matches(
            second,
            "aten.mm",
            "aten.matmul",
            "aten.bmm",
            "mm.default",
            "matmul.default",
            "bmm.default",
        ):
            accumulator_node = first
            matmul_node = second
            break
    if accumulator_node is None or matmul_node is None:
        return None
    accumulator = ctx.get_value(accumulator_node)
    matmul_args = list(normalized_aten_args(matmul_node))
    if accumulator is None or len(matmul_args) < 2:
        return None
    from ..lowering import resolve_contraction_operand

    lhs, transpose_lhs = resolve_contraction_operand(ctx, matmul_args[0])
    rhs, transpose_rhs = resolve_contraction_operand(ctx, matmul_args[1])
    if lhs is None or rhs is None:
        return None
    return emit_matmul_like(
        ctx,
        lhs,
        rhs,
        out=accumulator,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
    )


def lower_scalar_binary(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower an elementwise binary op whose second operand is a scalar value.

    Covers tile positions (``tile.begin`` and friends) and ``hl.grid`` indices,
    which reach codegen as plain ``index`` scalars rather than tensors.
    """
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..aten_lowering import normalized_aten_args

    named_ops = {
        "aten.add.Tensor": linalg_d.add,
        "aten.sub.Tensor": linalg_d.sub,
        "aten.mul.Tensor": linalg_d.mul,
        "aten.div.Tensor": linalg_d.div,
    }
    operation = next(
        (builder for name, builder in named_ops.items() if name in str(node.target)),
        None,
    )
    if operation is None:
        return None

    args = list(normalized_aten_args(node))
    if len(args) < 2 or (args[2] if len(args) > 2 else 1) != 1:
        return None

    values = [
        ctx.get_value(arg) if isinstance(arg, torch.fx.Node) else None for arg in args
    ]

    def as_tensor(value: object) -> ir.RankedTensorType | None:
        if value is None:
            return None
        try:
            return ir.RankedTensorType(value.type)
        except Exception:
            return None

    tensor_index = next(
        (i for i in (0, 1) if as_tensor(values[i]) is not None),
        None,
    )
    scalar_index = next(
        (i for i in (0, 1) if values[i] is not None and as_tensor(values[i]) is None),
        None,
    )
    if tensor_index is None or scalar_index is None:
        return None

    tensor_value = values[tensor_index]
    tensor_type = as_tensor(tensor_value)
    assert tensor_type is not None
    element_type = tensor_type.element_type

    scalar = ctx.cast_scalar_to(values[scalar_index], element_type)
    if scalar is None:
        return None

    shape = list(tensor_type.shape)
    filled = linalg_d.fill(scalar, outs=[tensor_d.EmptyOp(shape, element_type).result])
    lhs, rhs = (tensor_value, filled) if tensor_index == 0 else (filled, tensor_value)
    output = tensor_d.EmptyOp(shape, element_type).result
    return operation(lhs, rhs, outs=[output])
