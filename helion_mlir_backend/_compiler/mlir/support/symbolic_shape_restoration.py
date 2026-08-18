"""Restore symbolic and bounded metadata inside nested Helion loop bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    for graph_info in host_function.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op != "call_function"
                or getattr(node.target, "__name__", "") != "_for_loop"
            ):
                continue
            body_graph = host_function.device_ir.graphs[node.args[0]].graph
            placeholders = [
                item for item in body_graph.nodes if item.op == "placeholder"
            ]
            for placeholder, outer_node in zip(
                placeholders, node.args[3], strict=False
            ):
                _restore_placeholder_metadata(
                    placeholder,
                    outer_node,
                    body_graph,
                    node.args[2] if len(node.args) > 2 else None,
                    context,
                )


def _restore_placeholder_metadata(
    placeholder: torch.fx.Node,
    outer_node: object,
    body_graph: torch.fx.Graph,
    upper_bounds: object,
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

    _register_symbolic_shape_ids(shape_arg, outer_value, context, placeholder)
    concrete_shape = context.shape_from_nodes(list(shape_arg), "iter_arg")
    if isinstance(upper_bounds, (list, tuple)):
        for index, bound in enumerate(upper_bounds):
            if index >= len(concrete_shape):
                break
            try:
                concrete_shape[index] = min(concrete_shape[index], int(bound))
            except (TypeError, ValueError):
                continue
    concrete_value = torch.zeros(concrete_shape, dtype=outer_value.dtype)
    placeholder.meta["val"] = concrete_value
    _propagate_body_metadata(body_graph, placeholder, concrete_value, context)


def _register_symbolic_shape_ids(
    shape_arg: list | tuple,
    outer_value: torch.Tensor,
    context: BuildContext,
    placeholder: torch.fx.Node,
) -> None:
    import sympy

    for index, shape_node in enumerate(shape_arg):
        if not isinstance(shape_node, torch.fx.Node) or index >= len(outer_value.shape):
            continue
        if getattr(shape_node.target, "__name__", "") != "_get_symnode":
            continue
        key = shape_node.args[0] if shape_node.args else None
        block_id = block_id_from_key(key)
        if block_id is None:
            continue
        # Placeholder metadata is concretized below, so remember which block each
        # dimension came from while the symbols are still available.
        context.placeholder_dim_to_block_id[(id(placeholder), index)] = block_id
        dimension = outer_value.shape[index]
        expression = getattr(getattr(dimension, "node", None), "expr", None)
        if isinstance(dimension, torch.SymInt) and isinstance(expression, sympy.Symbol):
            context.block_symint_to_id[id(expression)] = block_id


def _propagate_new_var(
    body_graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
    value: torch.Tensor,
) -> None:
    for body_node in body_graph.nodes:
        if (
            body_node.op == "call_function"
            and getattr(body_node.target, "__name__", "") == "_new_var"
            and body_node.args
            and body_node.args[0] is placeholder
        ):
            body_node.meta["val"] = value


def _propagate_body_metadata(
    body_graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
    concrete_value: torch.Tensor,
    context: BuildContext,
) -> None:
    concrete_shape = list(concrete_value.shape)
    for body_node in body_graph.nodes:
        if body_node.op != "call_function":
            continue
        target_name = getattr(body_node.target, "__name__", "")
        if (
            target_name == "_new_var"
            and body_node.args
            and body_node.args[0] is placeholder
        ):
            body_node.meta["val"] = concrete_value
            continue
        if target_name in ("sym_size.int", "sym_size_int"):
            _propagate_sym_size(body_node, placeholder, concrete_shape)
            continue
        if target_name == "load":
            _propagate_load_shape(body_node, context)


def _propagate_sym_size(
    node: torch.fx.Node,
    placeholder: torch.fx.Node,
    concrete_shape: list[int],
) -> None:
    dimension = node.args[1] if len(node.args) > 1 else None
    if node.args and node.args[0] is placeholder and isinstance(dimension, int):
        if dimension < len(concrete_shape):
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
        return
