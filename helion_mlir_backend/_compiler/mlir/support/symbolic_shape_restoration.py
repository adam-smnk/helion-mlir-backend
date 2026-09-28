"""Restore symbolic and bounded metadata inside nested Helion loop bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
import torch

from .block_ids import block_id_from_key

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction

    from ..build_context import BuildContext


def restore_symbolic_shapes_in_bodies(
    host_function: HostFunction,
    context: BuildContext,
) -> None:
    """Copy outer loop metadata into nested body placeholders."""
    from ..analysis.geometry import is_loop_node

    for graph_info in host_function.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or not is_loop_node(node):
                continue
            body_graph = host_function.device_ir.graphs[node.args[0]].graph
            placeholders = [
                item for item in body_graph.nodes if item.op == "placeholder"
            ]
            for placeholder, outer_node in zip(
                placeholders, node.args[3], strict=False
            ):
                _restore_placeholder_metadata(
                    placeholder, outer_node, body_graph, context
                )


def _restore_placeholder_metadata(
    placeholder: torch.fx.Node,
    outer_node: object,
    body_graph: torch.fx.Graph,
    context: BuildContext,
) -> None:
    if not isinstance(outer_node, torch.fx.Node):
        return
    outer_value = outer_node.meta.get("val")
    if not isinstance(outer_value, torch.Tensor):
        return

    shape_arg = outer_node.args[0] if outer_node.args else None
    if not isinstance(shape_arg, (list, tuple)):
        placeholder.meta["val"] = outer_value
        _propagate_new_var(body_graph, placeholder, outer_value)
        return

    concrete_shape = context.shape_from_nodes(list(shape_arg), "iter_arg")
    concrete_value = torch.zeros(concrete_shape, dtype=outer_value.dtype)
    placeholder.meta["val"] = concrete_value
    _propagate_body_metadata(
        body_graph, placeholder, concrete_value, list(shape_arg), context
    )


def _propagate_new_var(
    body_graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
    value: torch.Tensor,
) -> None:
    for body_node in body_graph.nodes:
        if (
            body_node.op == "call_function"
            and body_node.target is tracing_ops._new_var
            and body_node.args
            and body_node.args[0] is placeholder
        ):
            body_node.meta["val"] = value


def _propagate_body_metadata(
    body_graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
    concrete_value: torch.Tensor,
    shape_arg: list,
    context: BuildContext,
) -> None:
    concrete_shape = list(concrete_value.shape)
    for body_node in body_graph.nodes:
        if body_node.op != "call_function":
            continue
        target = body_node.target
        if (
            target is tracing_ops._new_var
            and body_node.args
            and body_node.args[0] is placeholder
        ):
            body_node.meta["val"] = concrete_value
            continue
        if target is torch.ops.aten.sym_size.int:
            _propagate_sym_size(
                body_node, placeholder, concrete_shape, shape_arg, context
            )
            continue
        if target is memory_ops.load:
            _propagate_load_shape(body_node, context)


def _propagate_sym_size(
    node: torch.fx.Node,
    placeholder: torch.fx.Node,
    concrete_shape: list[int],
    shape_arg: list,
    context: BuildContext,
) -> None:
    dimension = node.args[1] if len(node.args) > 1 else None
    if not (node.args and node.args[0] is placeholder and isinstance(dimension, int)):
        return
    if dimension >= len(concrete_shape):
        return
    # Concretizing to a plain int below discards the dimension's symbol
    # identity, so tag the block id it came from using the same
    # ``tile_with_offset`` convention Helion's own device-IR pass uses (read
    # by ``resolve_index_descriptor``), instead of a side id()-keyed map.
    block_id = None
    previous = node.meta.get("val")
    if isinstance(previous, torch.SymInt):
        info = context.symbol_info(previous)
        if info is not None and info[1] == "block_size":
            block_id = info[0]
    if block_id is None and dimension < len(shape_arg):
        shape_node = shape_arg[dimension]
        if (
            isinstance(shape_node, torch.fx.Node)
            and shape_node.target is tracing_ops._get_symnode
            and shape_node.args
        ):
            block_id = block_id_from_key(shape_node.args[0])
    if block_id is not None:
        node.meta["tile_with_offset"] = {"block_id": block_id, "offset": 0}
    node.meta["val"] = concrete_shape[dimension]


def _propagate_load_shape(node: torch.fx.Node, context: BuildContext) -> None:
    indexes = node.args[1] if len(node.args) > 1 else None
    if indexes is None:
        return
    try:
        load_shape = context.shape_from_nodes(list(indexes), "load")
        old_value = node.meta.get("val")
        if isinstance(old_value, torch.Tensor) and len(load_shape) == len(
            old_value.shape
        ):
            node.meta["val"] = torch.zeros(load_shape, dtype=old_value.dtype)
    except Exception:
        # Best-effort shape refresh; on failure the node keeps its prior
        # (possibly stale but still usable) metadata.
        return
