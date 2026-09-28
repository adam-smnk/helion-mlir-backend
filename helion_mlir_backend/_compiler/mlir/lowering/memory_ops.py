"""Memory-related MLIR lowering helpers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
from mlir.dialects import linalg as linalg_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from ..analysis.tensor_effects import host_tensor_name
from ..support import NodeLoweringError
from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from . import emit
from .registry import lowers
from .slice_plan import SlicePlan
from .slice_plan import plan_slice

if TYPE_CHECKING:
    import torch.fx

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
    """Pass-through until boundary tiles are masked (plan Phase 6)."""
    return ctx.get_value(node.args[0])


@lowers(tracing_ops._inductor_lowering_extra)
def lower_inductor_extra(ctx: BuildContext, node: torch.fx.Node) -> None:
    """No value: an Inductor intermediate buffer of an ATen op (e.g. the sum of
    ``mean``) that only feeds that op's ``_extra_args``; the op's helper
    recomputes it."""
    return None


@lowers(memory_ops.store)
def lower_store(ctx: BuildContext, node: torch.fx.Node) -> None:
    """``insert_slice`` of the value into the destination's current SSA state."""
    name = host_tensor_name(node.args[0])
    if name is None or name not in ctx.tensors:
        raise NodeLoweringError(node, reason="the store target is not a host tensor")
    index_nodes, value_node = node.args[1], node.args[2]
    state = ctx.tensors.value(name)
    state_type = ir.RankedTensorType(state.type)
    plan = plan_slice(ctx, index_nodes, state_type, ctx.tensors.owned(name))
    if plan.gathers():
        raise UnsupportedOperationError(
            "store", reason="stores indexed by a tensor (scatter) are not supported"
        )
    value = _store_value(ctx, node, value_node, state_type.element_type, plan)
    rank = len(plan.dims)
    updated = tensor_d.InsertSliceOp(
        value,
        state,
        plan.offsets(),
        [],
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * rank,
        static_sizes=plan.static_sizes(),
        static_strides=[1] * rank,
    ).result
    ctx.tensors.rebind(name, updated)


def _store_value(
    ctx: BuildContext,
    node: torch.fx.Node,
    value_node: object,
    element_type: ir.Type,
    plan: SlicePlan,
) -> ir.Value:
    """The stored value as a tensor of the slice's shape and destination dtype."""
    value = ctx.get_value(value_node)
    if value is None:
        raise ValueNotFoundError(value_node, context="stored value")
    if not isinstance(value.type, ir.RankedTensorType):
        scalar = emit.cast_scalar(value, element_type)
        return linalg_d.fill(
            scalar, outs=[emit.empty(plan.value_shape(), element_type)]
        )
    shape = list(value.type.shape)
    if shape not in (plan.value_shape(), plan.static_sizes()):
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
