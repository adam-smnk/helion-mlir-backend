"""Registry for ATen operations with custom MLIR lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


_CUSTOM_TARGETS = (
    "aten.addmm",
    "aten.mm",
    "aten.matmul",
    "aten.bmm",
    "aten.baddbmm",
    "aten.permute",
    "aten.transpose",
    "aten.t.default",
    "aten._to_copy",
    "aten.to.",
    "convert_element_type",
)


def aten_target_matches(node: torch.fx.Node, *names: str) -> bool:
    """Match ATen targets by canonical name or FX overload short name."""
    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    return any(name in target_name or overload_name == name for name in names)


def is_custom_aten(node: torch.fx.Node) -> bool:
    """Return whether an ATen node is reserved for custom lowering."""
    return node.op == "call_function" and aten_target_matches(node, *_CUSTOM_TARGETS)


def lower_custom_aten(builder: object, node: torch.fx.Node) -> ir.Value | None:
    """Try the registered custom ATen lowerers in precedence order."""
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

    if aten_target_matches(node, "aten.add.Tensor", "add.Tensor"):
        lowered = lower_add_tensor(builder.context, node)
        if lowered is not None:
            return lowered

    if aten_target_matches(
        node,
        "aten.mm",
        "aten.matmul",
        "aten.bmm",
        "mm.default",
        "matmul.default",
        "bmm.default",
    ):
        lowered = builder._lower_aten_matmul(node)
        if lowered is not None:
            return lowered

    if aten_target_matches(node, "aten.relu", "relu.default"):
        lowered = lower_relu(builder.context, node)
        if lowered is not None:
            return lowered

    lowered = lower_dtype_convert(builder.context, node)
    if lowered is not None:
        return lowered

    lowered = lower_transpose_node(builder.context, node)
    if lowered is not None:
        return lowered

    lowered = lower_reduce_max_1d(builder.context, node)
    if lowered is not None:
        return lowered

    lowered = lower_passthrough(builder.context, node)
    if lowered is not None:
        return lowered

    return None


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


def lower_dtype_convert(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``.to(dtype)`` / ``_to_copy`` / ``convert_element_type`` on a tile."""
    import torch

    from ..aten_lowering import normalized_aten_args
    from ..support import torch_dtype_to_mlir

    if not aten_target_matches(
        node,
        "aten._to_copy",
        "aten.to.",
        "convert_element_type",
        "_to_copy.default",
        "to.dtype",
        "convert_element_type.default",
    ):
        return None

    args = list(normalized_aten_args(node))
    if not args or not isinstance(args[0], torch.fx.Node):
        return None
    source = ctx.get_value(args[0])
    if source is None:
        return None

    dtype = next((arg for arg in args[1:] if isinstance(arg, torch.dtype)), None)
    if dtype is None:
        dtype = node.kwargs.get("dtype")
    if dtype is None:
        value = node.meta.get("val")
        dtype = value.dtype if isinstance(value, torch.Tensor) else None
    if dtype is None:
        return None

    return convert_tensor_element_type(ctx, source, torch_dtype_to_mlir(dtype))


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


def lower_add_tensor(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``aten.add.Tensor`` with scalar and broadcast support."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if len(args) < 2 or (args[2] if len(args) > 2 else 1) != 1:
        return None
    lhs = ctx.get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
    rhs = ctx.get_value(args[1]) if isinstance(args[1], torch.fx.Node) else None

    def filled_add(tensor_value: ir.Value, scalar: float) -> ir.Value | None:
        tensor_type = ir.RankedTensorType(tensor_value.type)
        element_type = tensor_type.element_type
        if isinstance(element_type, ir.FloatType):
            attr = ir.FloatAttr.get(element_type, float(scalar))
        elif isinstance(element_type, ir.IntegerType):
            attr = ir.IntegerAttr.get(element_type, int(scalar))
        else:
            return None
        scalar_value = arith_d.ConstantOp(element_type, attr).result
        filled_empty = tensor_d.EmptyOp(list(tensor_type.shape), element_type).result
        filled = linalg_d.fill(scalar_value, outs=[filled_empty])
        output = tensor_d.EmptyOp(list(tensor_type.shape), element_type).result
        return linalg_d.add(tensor_value, filled, outs=[output])

    if lhs is not None and rhs is None and isinstance(args[1], (int, float)):
        return filled_add(lhs, args[1])
    if rhs is not None and lhs is None and isinstance(args[0], (int, float)):
        result = filled_add(rhs, args[0])
        if result is None:
            return None
        return result
    if lhs is None or rhs is None:
        return None

    lhs_type = ir.RankedTensorType(lhs.type)
    rhs_type = ir.RankedTensorType(rhs.type)
    lhs_shape = [int(dim) for dim in lhs_type.shape]
    rhs_shape = [int(dim) for dim in rhs_type.shape]
    output_rank = max(len(lhs_shape), len(rhs_shape))
    output_shape = [
        max(
            lhs_shape[-1 - index] if index < len(lhs_shape) else 1,
            rhs_shape[-1 - index] if index < len(rhs_shape) else 1,
        )
        for index in range(output_rank)
    ]
    output_shape.reverse()
    for index in range(output_rank):
        lhs_dim = lhs_shape[-1 - index] if index < len(lhs_shape) else 1
        rhs_dim = rhs_shape[-1 - index] if index < len(rhs_shape) else 1
        if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:
            return None

    lhs_element = lhs_type.element_type
    rhs_element = rhs_type.element_type
    if str(lhs_element) == str(rhs_element):
        output_element = lhs_element
    elif (
        isinstance(lhs_element, ir.IntegerType)
        and isinstance(rhs_element, ir.IntegerType)
    ) or (
        isinstance(lhs_element, ir.FloatType) and isinstance(rhs_element, ir.FloatType)
    ):
        output_element = (
            lhs_element if lhs_element.width >= rhs_element.width else rhs_element
        )
    elif isinstance(lhs_element, ir.FloatType) and isinstance(
        rhs_element, ir.IntegerType
    ):
        output_element = lhs_element
    elif isinstance(lhs_element, ir.IntegerType) and isinstance(
        rhs_element, ir.FloatType
    ):
        output_element = rhs_element
    else:
        return None

    def cast_scalar(
        value: ir.Value, source: ir.Type, target: ir.Type
    ) -> ir.Value | None:
        if str(source) == str(target):
            return value
        if isinstance(source, ir.IntegerType) and isinstance(target, ir.IntegerType):
            return (
                arith_d.ExtSIOp if source.width < target.width else arith_d.TruncIOp
            )(target, value).result
        if isinstance(source, ir.IntegerType) and isinstance(target, ir.FloatType):
            return arith_d.SIToFPOp(target, value).result
        if isinstance(source, ir.FloatType) and isinstance(target, ir.FloatType):
            return (
                arith_d.ExtFOp if source.width < target.width else arith_d.TruncFOp
            )(target, value).result
        return None

    if list(lhs_type.shape) != list(rhs_type.shape) or str(lhs_element) != str(
        rhs_element
    ):
        output_type = ir.RankedTensorType.get(output_shape, output_element)
        generated = tensor_d.GenerateOp(output_type, [])
        body = generated.operation.regions[0].blocks.append(
            *([ir.IndexType.get()] * len(output_shape))
        )
        with ir.InsertionPoint(body):
            indices = list(body.arguments)

            def operand_indices(shape: list[int]) -> list[ir.Value]:
                offset = len(output_shape) - len(shape)
                return [
                    ctx.index_const(0) if size == 1 else indices[offset + dimension]
                    for dimension, size in enumerate(shape)
                ]

            lhs_scalar = tensor_d.ExtractOp(
                lhs, operand_indices(lhs_shape), results=[lhs_element]
            ).result
            rhs_scalar = tensor_d.ExtractOp(
                rhs, operand_indices(rhs_shape), results=[rhs_element]
            ).result
            lhs_cast = cast_scalar(lhs_scalar, lhs_element, output_element)
            rhs_cast = cast_scalar(rhs_scalar, rhs_element, output_element)
            if lhs_cast is None or rhs_cast is None:
                return None
            summed = (
                arith_d.AddFOp(lhs_cast, rhs_cast).result
                if isinstance(output_element, ir.FloatType)
                else arith_d.AddIOp(lhs_cast, rhs_cast).result
            )
            tensor_d.YieldOp(summed)
        return generated.result

    output = tensor_d.EmptyOp(list(lhs_type.shape), lhs_element).result
    return linalg_d.add(lhs, rhs, outs=[output])


def lower_relu(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower floating-point ``aten.relu`` to a linalg max operation."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if not args or not isinstance(args[0], torch.fx.Node):
        return None
    input_value = ctx.get_value(args[0])
    if input_value is None:
        return None
    input_type = ir.RankedTensorType(input_value.type)
    if not isinstance(input_type.element_type, ir.FloatType):
        return None
    output = tensor_d.EmptyOp(list(input_type.shape), input_type.element_type).result
    zero_empty = tensor_d.EmptyOp(
        list(input_type.shape), input_type.element_type
    ).result
    zero = arith_d.ConstantOp(
        input_type.element_type,
        ir.FloatAttr.get(input_type.element_type, 0.0),
    ).result
    zero_tensor = linalg_d.fill(zero, outs=[zero_empty])
    return linalg_d.max(input_value, zero_tensor, outs=[output])


def lower_reduce_max_1d(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower a one-dimensional integer ``aten.max`` to a scalar tensor."""
    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    if "aten.max" not in target_name and overload_name != "max.default":
        return None
    from ..aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if not args or not isinstance(args[0], torch.fx.Node):
        return None
    input_value = ctx.get_value(args[0])
    if input_value is None:
        return None
    return lower_max_reduce_from_tensor(ctx, input_value)


def lower_max_reduce_from_tensor(
    ctx: BuildContext, input_value: ir.Value
) -> ir.Value | None:
    """Lower a rank-one integer tensor maximum reduction."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    input_type = ir.RankedTensorType(input_value.type)
    if input_type.rank != 1 or not isinstance(input_type.element_type, ir.IntegerType):
        return None
    element_type = input_type.element_type
    extent = int(input_type.shape[0])
    if extent <= 0:
        return None
    minimum = -(1 << (element_type.width - 1))
    initial = arith_d.ConstantOp(
        element_type,
        ir.IntegerAttr.get(element_type, minimum),
    ).result
    loop = scf_d.ForOp(
        ctx.index_const(0),
        ctx.index_const(extent),
        ctx.index_const(1),
        iter_args=[initial],
    )
    with ir.InsertionPoint(loop.body):
        induction_variable = loop.body.arguments[0]
        accumulator = loop.body.arguments[1]
        current = tensor_d.ExtractOp(
            input_value,
            [induction_variable],
            results=[element_type],
        ).result
        scf_d.YieldOp([arith_d.MaxSIOp(current, accumulator).result])
    scalar_empty = tensor_d.EmptyOp([], element_type).result
    return tensor_d.InsertOp(loop.results[0], scalar_empty, []).result
