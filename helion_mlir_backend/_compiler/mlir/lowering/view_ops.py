"""Static view/reshape lowering without an ATen helper round-trip (keeps shapes static)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    source = node.args[0] if node.args else None
    value = ctx.get_value(source) if isinstance(source, torch.fx.Node) else None
    result_shape = ctx.shape_from_node_meta(node)
    if value is None or not result_shape:
        return NOT_APPLICABLE
    reshaped = static_reshape(value, result_shape)
    return NOT_APPLICABLE if reshaped is None else reshaped


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
