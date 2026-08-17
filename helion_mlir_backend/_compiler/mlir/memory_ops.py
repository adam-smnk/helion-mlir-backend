"""Memory-related MLIR lowering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch.fx


def lower_getitem(builder: object, node: torch.fx.Node) -> ir.Value | None:
    """Extract one result from an ``scf.for`` result container."""
    container_value = builder._get_value(node.args[0])
    if container_value is None:
        return None
    index = int(node.args[1])
    if hasattr(container_value, "results"):
        return container_value.results[index]
    return container_value


def lower_store(builder: object, node: torch.fx.Node) -> None:
    """Record or apply a Helion store in the active loop context."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    index_nodes = node.args[1]
    value_node = node.args[2]
    value = builder._get_value(value_node)
    assert value is not None, f"No value for store value node {value_node}"
    ndim = len(ir.RankedTensorType(value.type).shape)

    if builder._for_store_ctx_stack:
        context = builder._for_store_ctx_stack[-1]
        current = context.get("current")
        block_id = int(context.get("block_id", -1))
        inner_dim = int(context.get("inner_dim", -1))
        rank = int(context.get("rank", ndim))
        if (
            current is not None
            and 0 <= inner_dim < rank
            and block_id in builder._block_id_to_iv
        ):
            source_type = ir.RankedTensorType(value.type)
            updated = tensor_d.InsertSliceOp(
                value,
                current,
                [
                    builder._block_id_to_iv[block_id]
                    if dimension == inner_dim
                    else builder._get_index_const(0)
                    for dimension in range(rank)
                ],
                [],
                [],
                static_offsets=[ir.ShapedType.get_dynamic_size()] * rank,
                static_sizes=list(source_type.shape),
                static_strides=[1] * rank,
            ).result
            context["current"] = updated
            return

    sym_to_block_id = builder._build_sym_to_block_id()
    offsets: list[ir.Value] = []
    for dimension, index_node in enumerate(index_nodes):
        if dimension >= ndim:
            break
        block_id = builder._infer_block_id_from_index(index_node, sym_to_block_id)
        if block_id is not None and block_id in builder._block_id_to_iv:
            offsets.append(builder._block_id_to_iv[block_id])
        else:
            offsets.append(builder._get_index_const(0))
    builder._forall_insert_slices.append((value, offsets))
