"""Memory-related MLIR lowering helpers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
from mlir.dialects import linalg as linalg_d
import mlir.ir as ir
import torch

from ..analysis.tensor_effects import host_tensor_name
from ..aten_bridge import call_helper
from ..support import NodeLoweringError
from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from . import emit
from .load_slice_ops import load_tile
from .registry import lowers
from .slice_plan import SlicePlan
from .slice_plan import plan_slice
from .view_ops import static_reshape

if TYPE_CHECKING:
    from ..build_context import BuildContext


@lowers(operator.getitem)
def lower_getitem(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Extract one result from an ``scf.for`` result container."""
    container_value = ctx.get_value(node.args[0])
    if container_value is None:
        return None
    index = int(node.args[1])
    if hasattr(container_value, "results"):
        return container_value.results[index]
    return container_value


@lowers(tracing_ops._mask_to)
def lower_mask_to(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``_mask_to(x, other)``: ``other`` outside the loop along each tile dim of ``x``.

    A dim belongs to a tile through its size's symbol; only loops with a partial
    last tile need a mask.
    """
    source, other = node.args
    value = ctx.get_value(source)
    bounds = {}
    for dim, size in enumerate(source.meta["val"].shape):
        block_id = ctx.env.resolve_block_id(size)
        valid = ctx.block_id_to_valid.get(block_id)
        if valid is not None:
            bounds[dim] = ctx.as_index(valid)
    return emit.mask(value, bounds, other) if bounds else value


@lowers(tracing_ops._inductor_lowering_extra)
def lower_inductor_extra(ctx: BuildContext, node: torch.fx.Node) -> None:
    """No value: an Inductor intermediate buffer of an ATen op (e.g. the sum of
    ``mean``) that only feeds that op's ``_extra_args``; the op's helper
    recomputes it."""
    return None


@lowers(memory_ops.store)
def lower_store(ctx: BuildContext, node: torch.fx.Node) -> None:
    """``hl.store(tensor, index, value, extra_mask)``: an ``insert_slice`` into the
    tensor's current value, of the tile's real part where ``extra_mask`` holds."""
    name = host_tensor_name(node.args[0])
    if name is None or name not in ctx.tensors:
        raise NodeLoweringError(node, reason="the store target is not a host tensor")
    index_nodes, value_node = node.args[1], node.args[2]
    extra_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("extra_mask")
    state = ctx.tensors.value(name)
    plan = plan_slice(ctx, index_nodes, state, ctx.tensors.owned(name))
    if plan.gathers():
        raise UnsupportedOperationError(
            "store", reason="stores indexed by a tensor (scatter) are not supported"
        )
    element_type = ir.RankedTensorType(state.type).element_type
    value = _store_value(ctx, node, value_node, element_type, plan)
    if extra_mask is not None or plan.is_partial():
        value = static_reshape(value, plan.value_shape())
    if extra_mask is not None:
        current = load_tile(state, plan)
        value = call_helper(
            ctx, node, torch.ops.aten.where.self, (extra_mask, value, current), {}
        )
    if plan.is_partial():
        zeros = [ctx.index_const(0)] * len(plan.value_shape())
        value = emit.extract_slice(value, zeros, plan.value_sizes())
    ctx.tensors.rebind(
        name, emit.insert_slice(value, state, plan.offsets(), plan.sizes())
    )


def _store_value(
    ctx: BuildContext,
    node: torch.fx.Node,
    value_node: object,
    element_type: ir.Type,
    plan: SlicePlan,
) -> ir.Value:
    """The stored value as a tile of the plan's shape (with or without the reduced
    dims) and the destination dtype."""
    value = ctx.get_value(value_node)
    if value is None:
        raise ValueNotFoundError(value_node, context="stored value")
    if not isinstance(value.type, ir.RankedTensorType):
        scalar = emit.cast_scalar(value, element_type)
        return linalg_d.fill(
            scalar, outs=[emit.empty(plan.value_shape(), element_type)]
        )
    shape = list(value.type.shape)
    if shape not in (plan.value_shape(), plan.tile_shape()):
        raise UnsupportedOperationError(
            "store with transposed or mismatched tile layout",
            reason=(
                f"storing a tile of shape {shape} into a slice of shape "
                f"{plan.value_shape()}; the stored value's tile order does not "
                "match the order the destination is indexed"
            ),
            alternatives=[
                "reorder explicitly, e.g. out[a, b] = src[b, a].permute(1, 0)",
                "index the destination in the same order the value is loaded",
            ],
        )
    return emit.cast_tensor(value, element_type)
