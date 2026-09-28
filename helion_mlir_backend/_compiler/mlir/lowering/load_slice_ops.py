"""Ordinary Helion tile loads lowered to tensor.extract_slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.memory_ops as memory_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from ..analysis.tensor_effects import host_tensor_name
from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from .registry import lowers
from .subscript_ops import gather

if TYPE_CHECKING:
    import torch.fx

    from ..build_context import BuildContext


@lowers(memory_ops.load)
def lower_load(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a Helion load to a static tensor extract slice."""

    tensor_node = node.args[0]
    index_nodes = node.args[1]
    name = host_tensor_name(tensor_node)
    if name is not None and name in ctx.tensors:
        tensor_value = ctx.tensors.value(name)
        owned = ctx.tensors.owned(name)
    else:
        tensor_value = ctx.get_value(tensor_node)
        owned = {}
    if tensor_value is None:
        raise ValueNotFoundError(tensor_node, context="loaded tensor")
    tensor_type = ir.RankedTensorType(tensor_value.type)

    # Build authoritative slice plan from index metadata.
    from .slice_plan import plan_slice

    plan = plan_slice(ctx, index_nodes, tensor_type, owned)

    # Extract at full rank (no rank reduction at the op level): letting MLIR
    # infer which size-1 dims to drop from static_sizes alone is ambiguous
    # whenever a *kept* (non-reduced) dim also happens to have extent 1 (a
    # tile whose block size is 1), which can trigger a native assertion.
    # Instead, always keep every dim here, then explicitly collapse only the
    # scalar-indexed (``reduces``) dims via a reassociation map, which is
    # unambiguous because it names dims by position, not by size.
    full_shape = plan.static_sizes()
    full_type = ir.RankedTensorType.get(full_shape, tensor_type.element_type)
    loaded = tensor_d.ExtractSliceOp(
        full_type,
        tensor_value,
        plan.offsets(),
        [],
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * len(plan.dims),
        static_sizes=full_shape,
        static_strides=[1] * len(plan.dims),
    ).result

    reduced_dims = set(plan.reduced_dims())
    if reduced_dims:
        result_shape = plan.value_shape()
        if not result_shape:
            result_shape = [1]
        result_type = ir.RankedTensorType.get(result_shape, tensor_type.element_type)
        reassociation = _collapse_reassociation(len(full_shape), reduced_dims)
        loaded = tensor_d.CollapseShapeOp(result_type, loaded, reassociation).result

    gathers = plan.gathers()
    if not gathers:
        return loaded
    # Helion and torch agree on the result shape for one index tensor that is
    # 1-D or indexes a 1-D tensor; they differ beyond that.
    (dimension, index), *others = gathers
    if others or (len(plan.dims) > 1 and ir.RankedTensorType(index.type).rank != 1):
        raise UnsupportedOperationError(
            "load", reason="only one 1-D index tensor per load is supported"
        )
    position = dimension - sum(1 for reduced in reduced_dims if reduced < dimension)
    return gather(ctx, node, loaded, [slice(None)] * position + [index])


def _collapse_reassociation(rank: int, reduced_dims: set[int]) -> list[list[int]]:
    """Build a ``tensor.collapse_shape`` reassociation dropping ``reduced_dims``.

    Each reduced (guaranteed extent-1) dim is merged into the nearest kept
    dim's group, preferring the next kept dim to its right, falling back to
    the previous one. Unambiguous by construction (explicit index grouping,
    not size-based inference).
    """
    kept = [d for d in range(rank) if d not in reduced_dims]
    if not kept:
        return [list(range(rank))]
    groups: dict[int, list[int]] = {k: [k] for k in kept}
    for d in range(rank):
        if d not in reduced_dims:
            continue
        target = next((k for k in kept if k > d), None)
        if target is None:
            target = max(k for k in kept if k < d)
        groups[target].append(d)
    return [sorted(groups[k]) for k in sorted(groups)]
