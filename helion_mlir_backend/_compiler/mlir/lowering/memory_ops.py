"""Memory-related MLIR lowering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch.fx

    from ..build_context import BuildContext


def lower_getitem(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Extract one result from an ``scf.for`` result container."""
    container_value = ctx.get_value(node.args[0])
    if container_value is None:
        return None
    index = int(node.args[1])
    if hasattr(container_value, "results"):
        return container_value.results[index]
    return container_value


def _cast_store_value(ctx: BuildContext, value: ir.Value, target: ir.Value) -> ir.Value:
    """Convert a stored tile to the destination element type when they differ."""
    import mlir.ir as ir

    from ..aten_bridge import convert_tensor_element_type

    try:
        value_type = ir.RankedTensorType(value.type)
        target_type = ir.RankedTensorType(target.type)
    except Exception:
        return value
    if str(value_type.element_type) == str(target_type.element_type):
        return value
    converted = convert_tensor_element_type(ctx, value, target_type.element_type)
    return converted if converted is not None else value


def lower_store(ctx: BuildContext, node: torch.fx.Node) -> None:
    """Record or apply a Helion store in the active loop context."""
    index_nodes = node.args[1]
    value_node = node.args[2]
    value = ctx.get_value(value_node)
    assert value is not None, f"No value for store value node {value_node}"

    target_value = ctx.get_value(node.args[0])
    if target_value is not None:
        value = _cast_store_value(ctx, value, target_value)

    if ctx.for_store_ctx_stack:
        _store_into_synthetic_accumulator(ctx, index_nodes, value)
        return

    # Rare: the destination already has an SSA value bound (its dimensions
    # can be read directly from the value's own type).
    if target_value is not None and _store_via_bound_target(
        ctx, index_nodes, value, target_value, node
    ):
        return

    # Common case: the output tensor is created later in build_kernel_body,
    # so target_value has no SSA value yet and offsets must be inferred
    # positionally against the stored value's own shape.
    _store_via_deferred_target(ctx, index_nodes, value, target_value, node)


def _store_into_synthetic_accumulator(
    ctx: BuildContext, index_nodes: list | tuple, value: ir.Value
) -> None:
    """Insert into the active loop level's synthetic per-iteration accumulator."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    context = ctx.for_store_ctx_stack[-1]
    current = context.current

    # Compute the descriptor-based per-iteration insert plan once, on
    # first use.
    store_plan = context.store_plan
    if store_plan is None and current is not None:
        from .slice_plan import plan_slice

        target_type = ir.RankedTensorType(current.type)
        store_plan = plan_slice(ctx, index_nodes, target_type)
        context.store_plan = store_plan

    if store_plan is not None and current is not None:
        offsets = store_plan.offsets()
        static_sizes = store_plan.static_sizes()
        updated = tensor_d.InsertSliceOp(
            value,
            current,
            offsets,
            [],
            [],
            static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
            static_sizes=static_sizes,
            static_strides=[1] * len(offsets),
        ).result
        context.current = updated


def _store_via_bound_target(
    ctx: BuildContext,
    index_nodes: list | tuple,
    value: ir.Value,
    target_value: ir.Value,
    node: torch.fx.Node,
) -> bool:
    """Try the descriptor-based terminal store; return False to defer."""
    import mlir.ir as ir

    from ..support.errors import NodeLoweringError

    try:
        from .slice_plan import plan_slice

        target_type = ir.RankedTensorType(target_value.type)
        store_plan = plan_slice(ctx, index_nodes, target_type)
        offsets = store_plan.offsets()
        static_sizes = store_plan.static_sizes()
        target_tensor_id = _target_tensor_id(node)
        ctx.forall_insert_slices.append(
            (value, offsets, static_sizes, target_tensor_id)
        )
        return True
    except NodeLoweringError:
        # Expected bail signal: a tile index has no resolvable block id
        # (e.g. target_value's dimension doesn't match index_nodes yet).
        # Any other exception is a real bug and must propagate.
        return False


def _store_via_deferred_target(
    ctx: BuildContext,
    index_nodes: list | tuple,
    value: ir.Value,
    target_value: ir.Value | None,
    node: torch.fx.Node,
) -> None:
    """Positional terminal store used when the destination has no SSA value yet."""
    import mlir.ir as ir

    offsets: list[ir.Value] = []
    static_sizes: list[int] = []
    value_shape = list(ir.RankedTensorType(value.type).shape)
    value_dim = 0
    target_rank = len(index_nodes)
    if target_value is not None:
        target_rank = max(
            len(index_nodes), len(ir.RankedTensorType(target_value.type).shape)
        )
    else:
        target_shape = ctx.shape_from_node_meta(node.args[0])
        if target_shape is not None:
            target_rank = max(target_rank, len(target_shape))
    for index_node in index_nodes:
        if ctx.is_scalar_index_node(index_node):
            scalar_offset = ctx.get_value(index_node)
            offsets.append(
                ctx.cast_to_index(scalar_offset)
                if scalar_offset is not None
                else ctx.index_const(0)
            )
            static_sizes.append(1)
            continue

        if isinstance(index_node, slice):
            offsets.append(ctx.index_const(0))
            if value_dim < len(value_shape):
                static_sizes.append(value_shape[value_dim])
                value_dim += 1
            else:
                static_sizes.append(1)
            continue

        if value_dim >= len(value_shape):
            break
        block_id = ctx.infer_block_id_from_index(index_node)
        if block_id is not None and block_id in ctx.block_id_to_iv:
            offsets.append(ctx.block_id_to_iv[block_id])
        else:
            offsets.append(ctx.index_const(0))
        static_sizes.append(value_shape[value_dim])
        value_dim += 1

    if len(offsets) < target_rank:
        offsets.extend(ctx.index_const(0) for _ in range(target_rank - len(offsets)))
        static_sizes.extend([1] * (target_rank - len(static_sizes)))
    elif len(offsets) > target_rank:
        offsets = offsets[:target_rank]
        static_sizes = static_sizes[:target_rank]

    if (
        target_rank > len(value_shape)
        and len(value_shape) + (target_rank - len(value_shape)) == target_rank
    ):
        reduction = target_rank - len(value_shape)
        static_sizes = [1] * reduction + value_shape

    ctx.forall_insert_slices.append(
        (value, offsets, static_sizes, _target_tensor_id(node))
    )


def _target_tensor_id(node: torch.fx.Node) -> int | None:
    """``id()`` of the store's destination FakeTensor, when resolvable."""
    import torch

    target_node = node.args[0]
    if not isinstance(target_node, torch.fx.Node):
        return None
    target_val = target_node.meta.get("val")
    return id(target_val) if isinstance(target_val, torch.Tensor) else None
