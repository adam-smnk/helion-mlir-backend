"""Static views and reshapes without an ATen helper round-trip (keeps shapes
static), and ``hl.split``/``hl.join``."""

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
    from ..build_context import BuildContext

aten = torch.ops.aten


@lowers(aten.view.default, aten.reshape.default)
def lower_static_reshape(ctx: BuildContext, node: torch.fx.Node) -> object:
    from ..aten_bridge import infer_results

    value = ctx.get_value(node.args[0])
    (result,) = infer_results(ctx, node)
    reshaped = static_reshape(value, [int(dim) for dim in result.shape])
    return NOT_APPLICABLE if reshaped is None else reshaped


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
    joined = emit.empty([*value_type.shape, 2], value_type.element_type)
    for half, value in enumerate(halves):
        joined = put(ctx, joined, value, rank, ctx.index_const(half))
    return joined


def take(ctx: BuildContext, value: ir.Value, dim: int, position: ir.Value) -> ir.Value:
    """``value`` at ``position`` along ``dim``, without that dimension."""
    shape = list(ir.RankedTensorType(value.type).shape)
    offsets = [ctx.index_const(0)] * len(shape)
    offsets[dim] = position
    sizes = [*shape[:dim], 1, *shape[dim + 1 :]]
    sliced = emit.extract_slice(value, offsets, sizes)
    return static_reshape(sliced, shape[:dim] + shape[dim + 1 :])


def put(
    ctx: BuildContext, dest: ir.Value, item: ir.Value, dim: int, position: ir.Value
) -> ir.Value:
    """``dest`` with ``item`` stored at ``position`` along ``dim``."""
    shape = list(ir.RankedTensorType(dest.type).shape)
    offsets = [ctx.index_const(0)] * len(shape)
    offsets[dim] = position
    sizes = [*shape[:dim], 1, *shape[dim + 1 :]]
    return emit.insert_slice(static_reshape(item, sizes), dest, offsets, sizes)


def static_reshape(value: ir.Value, result_shape: list[int]) -> ir.Value | None:
    """``tensor.reshape`` to a static shape with the same element count, else ``None``."""
    source_type = ir.RankedTensorType(value.type)
    if list(source_type.shape) == list(result_shape):
        return value
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


def _numel(shape: object) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product
