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

    # Compute the result shape (scalar-indexed dimensions are dropped).
    result_shape = plan.value_shape()
    if not result_shape:
        result_shape = [1]

    result_type = ir.RankedTensorType.get(result_shape, tensor_type.element_type)
    return tensor_d.ExtractSliceOp(
        result_type,
        tensor_value,
        plan.offsets(),
        [],
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * len(plan.dims),
        static_sizes=plan.static_sizes(),
        static_strides=[1] * len(plan.dims),
    ).result
