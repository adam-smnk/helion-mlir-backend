"""Subscripts of device tensors: slices, new axes, scalar indices and gathers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.view_ops as view_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch
import torch.fx

from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from . import emit
from .registry import lowers
from .view_ops import static_reshape

if TYPE_CHECKING:
    from ..build_context import BuildContext


@lowers(view_ops.subscript)
def lower_subscript(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``tensor[index]``: an extract slice and reshape, or a gather for tensor indices.

    ``index`` holds full slices, ``None`` (new axes), scalar positions and index
    tensors; a gather goes through the ``aten.index.Tensor`` helper.
    """
    source_node, index = node.args[:2]
    source = ctx.get_value(source_node)
    if source is None:
        raise ValueNotFoundError(source_node, context="subscripted tensor")
    items = [_index_item(ctx, item) for item in index]
    if any(_is_index_tensor(item) for item in items):
        return gather(ctx, node, source, items)

    source_type = ir.RankedTensorType(source.type)
    dims = iter(source_type.shape)
    offsets: list[ir.Value] = []
    sizes: list[int] = []
    result_shape: list[int] = []
    for item in items:
        if item is None:
            result_shape.append(1)
            continue
        size = next(dims, None)
        if size is None:
            raise UnsupportedOperationError("subscript", reason="too many indices")
        if _is_full_slice(item):
            offsets.append(ctx.index_const(0))
            sizes.append(size)
            result_shape.append(size)
        elif isinstance(item, ir.Value) or (isinstance(item, int) and 0 <= item < size):
            offsets.append(ctx.as_index(item))
            sizes.append(1)
        else:
            raise UnsupportedOperationError("subscript", reason=f"index {item!r}")
    for size in dims:
        offsets.append(ctx.index_const(0))
        sizes.append(size)
        result_shape.append(size)

    if sizes != list(source_type.shape):
        source = tensor_d.ExtractSliceOp(
            ir.RankedTensorType.get(sizes, source_type.element_type),
            source,
            offsets,
            [],
            [],
            static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
            static_sizes=sizes,
            static_strides=[1] * len(offsets),
        ).result
    reshaped = static_reshape(source, result_shape)
    assert reshaped is not None
    return reshaped


def gather(
    ctx: BuildContext, node: torch.fx.Node, source: ir.Value, items: list[object]
) -> ir.Value:
    """``source[items]`` with index tensors and full slices, via ``aten.index.Tensor``.

    Index tensors are widened to ``i64``: torch-mlir fails on narrower ones.
    """
    from ..aten_bridge import call_helper

    i64 = ir.IntegerType.get_signless(64)
    indices: list[ir.Value | None] = []
    for item in items:
        if _is_full_slice(item):
            indices.append(None)
        elif _is_index_tensor(item):
            indices.append(emit.cast_tensor(item, i64))
        else:
            raise UnsupportedOperationError(
                "gather", reason=f"index {item!r} next to an index tensor"
            )
    while indices and indices[-1] is None:
        indices.pop()
    return call_helper(ctx, node, torch.ops.aten.index.Tensor, (source, indices), {})


def _index_item(ctx: BuildContext, item: object) -> object:
    if not isinstance(item, torch.fx.Node):
        return item
    value = ctx.get_value(item)
    if value is None:
        raise ValueNotFoundError(item, context="subscript index")
    return value


def _is_index_tensor(item: object) -> bool:
    return isinstance(item, ir.Value) and (
        isinstance(item.type, ir.RankedTensorType) and item.type.rank > 0
    )


def _is_full_slice(item: object) -> bool:
    return isinstance(item, slice) and item == slice(None)
