"""Views and reshapes without an ATen helper round-trip, and ``hl.split``/``hl.join``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.view_ops as helion_view_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch

from . import emit
from .registry import NOT_APPLICABLE
from .registry import lowers

if TYPE_CHECKING:
    import sympy

    from ..build_context import BuildContext

aten = torch.ops.aten


@lowers(aten.view.default, aten.reshape.default)
def lower_view(ctx: BuildContext, node: torch.fx.Node) -> object:
    reshaped = view(ctx, node)
    return NOT_APPLICABLE if reshaped is None else reshaped


def view(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``node``'s view or reshape of its first argument without a helper, else ``None``.

    With runtime (``?``) dims only unit dims may be added or removed, and since
    ``?`` types cannot tell sizes apart, the other dims' size symbols must match.
    """
    from ..aten_bridge import infer_results
    from ..aten_bridge.helpers import static_dim

    source = node.args[0]
    value = ctx.get_value(source)
    (result,) = infer_results(ctx, node)
    shape = [static_dim(dim) for dim in result.shape]
    dynamic = any(
        ir.ShapedType.is_dynamic_size(dim)
        for dim in [*ir.RankedTensorType(value.type).shape, *shape]
    )
    if dynamic and _non_unit_sizes(ctx, source) != _non_unit_sizes(ctx, node):
        return None
    return reshape(value, shape)


def _non_unit_sizes(ctx: BuildContext, node: torch.fx.Node) -> list[sympy.Expr]:
    sizes = (ctx.sizes.expr(size) for size in node.meta["val"].shape)
    return [size for size in sizes if size != 1]


@lowers(helion_view_ops.split)
def lower_split(ctx: BuildContext, node: torch.fx.Node) -> emit.Results:
    """``hl.split(x)``: the two halves of ``x``'s last dimension (of size 2)."""
    value = ctx.get_value(node.args[0])
    last = ir.RankedTensorType(value.type).rank - 1
    return emit.Results(
        [take(ctx, value, last, ctx.index_const(half)) for half in (0, 1)]
    )


@lowers(helion_view_ops.join)
def lower_join(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``hl.join(a, b)``: ``a`` and ``b`` stacked along a new last dimension."""
    halves = [ctx.get_value(arg) for arg in node.args[:2]]
    value_type = ir.RankedTensorType(halves[0].type)
    rank = value_type.rank
    joined = emit.empty([*emit.sizes(halves[0]), 2], value_type.element_type)
    for half, value in enumerate(halves):
        joined = put(ctx, joined, value, rank, ctx.index_const(half))
    return joined


def take(ctx: BuildContext, value: ir.Value, dim: int, position: ir.Value) -> ir.Value:
    """``value`` at ``position`` along ``dim``, without that dimension."""
    shape = list(ir.RankedTensorType(value.type).shape)
    offsets = [ctx.index_const(0)] * len(shape)
    offsets[dim] = position
    sizes = emit.sizes(value)
    sizes[dim] = 1
    return emit.extract_slice(value, offsets, sizes, shape[:dim] + shape[dim + 1 :])


def put(
    ctx: BuildContext, dest: ir.Value, item: ir.Value, dim: int, position: ir.Value
) -> ir.Value:
    """``dest`` with ``item`` stored at ``position`` along ``dim``."""
    shape = list(ir.RankedTensorType(dest.type).shape)
    offsets = [ctx.index_const(0)] * len(shape)
    offsets[dim] = position
    sizes = emit.sizes(dest)
    sizes[dim] = 1
    shape[dim] = 1
    # Not rank-reducing: that trips an MLIR assertion (areEquivalentSlices) in the opt pipeline.
    return emit.insert_slice(reshape(item, shape), dest, offsets, sizes)


def reshape(value: ir.Value, result_shape: list[int]) -> ir.Value | None:
    """``value`` reshaped without a helper, else ``None``: a ``tensor.reshape`` for
    static shapes; with runtime (``?``) dims, only unit dims may be added or
    removed, and the other dims keep their sizes (the caller knows they do)."""
    source_type = ir.RankedTensorType(value.type)
    if list(source_type.shape) == list(result_shape):
        return value
    if any(
        ir.ShapedType.is_dynamic_size(dim)
        for dim in [*source_type.shape, *result_shape]
    ):
        return _unit_dim_reshape(value, list(result_shape))
    if any(dim <= 0 for dim in result_shape) or _numel(source_type.shape) != _numel(
        result_shape
    ):
        return None
    i32 = ir.IntegerType.get_signless(32)
    shape = tensor_d.FromElementsOp(
        ir.RankedTensorType.get([len(result_shape)], i32),
        [emit.constant(i32, dim) for dim in result_shape],
    ).result
    result_type = ir.RankedTensorType.get(result_shape, source_type.element_type)
    return tensor_d.ReshapeOp(result_type, value, shape).result


def _unit_dim_reshape(value: ir.Value, result_shape: list[int]) -> ir.Value | None:
    """Collapse ``value``'s unit dims, then expand to ``result_shape``'s, if the
    other dims agree in order."""
    source_type = ir.RankedTensorType(value.type)
    source_shape = list(source_type.shape)
    core = [dim for dim in source_shape if dim != 1]
    if core != [dim for dim in result_shape if dim != 1]:
        return None
    element_type = source_type.element_type
    core_sizes = [
        size
        for size, dim in zip(emit.sizes(value), source_shape, strict=True)
        if dim != 1
    ]
    if len(core) != len(source_shape):
        value = tensor_d.CollapseShapeOp(
            ir.RankedTensorType.get(core, element_type),
            value,
            _unit_groups(source_shape),
        ).result
    if len(core) == len(result_shape):
        return value
    remaining = iter(core_sizes)
    output = [1 if dim == 1 else next(remaining) for dim in result_shape]
    return tensor_d.ExpandShapeOp(
        ir.RankedTensorType.get(result_shape, element_type),
        value,
        _unit_groups(result_shape),
        [size for size in output if isinstance(size, ir.Value)],
        result_shape,
    ).result


def _unit_groups(shape: list[int]) -> list[list[int]]:
    return emit.reassociation(
        len(shape), [dim for dim, size in enumerate(shape) if size == 1]
    )


def _numel(shape: object) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product
