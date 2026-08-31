"""Ordinary Helion tile loads lowered to tensor.extract_slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch.fx

    from ..build_context import BuildContext


def lower_load(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a Helion load to a static tensor extract slice."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    tensor_node = node.args[0]
    index_nodes = node.args[1]
    tensor_value = ctx.get_value(tensor_node)
    assert tensor_value is not None, f"No value for tensor node {tensor_node}"
    tensor_type = ir.RankedTensorType(tensor_value.type)
    ndim = len(tensor_type.shape)

    # Fast path: 1-D gather via lower_flat_gather.
    if ndim == 1 and len(index_nodes) == 1:
        gather_index_value = ctx.get_value(index_nodes[0])
        if gather_index_value is not None:
            try:
                gather_index_type = ir.RankedTensorType(gather_index_value.type)
            except Exception:
                gather_index_type = None
            if gather_index_type is not None and gather_index_type.rank >= 1:
                from .load_ops import lower_flat_gather

                gathered = lower_flat_gather(
                    ctx,
                    tensor_node,
                    tensor_value,
                    gather_index_value,
                    gather_index_type,
                    tensor_type,
                )
                if gathered is not None:
                    return gathered

    # Build authoritative slice plan from index metadata.
    from .slice_plan import plan_slice

    plan = plan_slice(ctx, index_nodes, tensor_type)

    # Extract at full rank (no rank reduction at the op level): letting MLIR
    # infer which size-1 dims to drop from static_sizes alone is ambiguous
    # whenever a *kept* (non-reduced) dim also happens to have extent 1 (a
    # tile whose block size is 1), which can trigger a native assertion.
    # Instead, always keep every dim here, then explicitly collapse only the
    # scalar-indexed (``reduces``) dims via a reassociation map, which is
    # unambiguous because it names dims by position, not by size.
    full_shape = plan.static_sizes()
    full_type = ir.RankedTensorType.get(full_shape, tensor_type.element_type)
    extracted = tensor_d.ExtractSliceOp(
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
    if not reduced_dims:
        return extracted

    result_shape = plan.value_shape()
    if not result_shape:
        result_shape = [1]
    result_type = ir.RankedTensorType.get(result_shape, tensor_type.element_type)
    reassociation = _collapse_reassociation(len(full_shape), reduced_dims)
    return tensor_d.CollapseShapeOp(result_type, extracted, reassociation).result


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
