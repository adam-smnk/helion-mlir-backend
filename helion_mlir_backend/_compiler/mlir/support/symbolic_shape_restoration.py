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
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "_for_loop":
                continue

            body_graph_id: int = node.args[0]
            iter_arg_outer_nodes = list(node.args[3])
            body_graph = host_function.device_ir.graphs[body_graph_id].graph
            body_placeholders = [
                body_node
                for body_node in body_graph.nodes
                if body_node.op == "placeholder"
            ]
            for placeholder, outer_node in zip(
                body_placeholders, iter_arg_outer_nodes, strict=False
            ):
                if not isinstance(outer_node, torch.fx.Node):
                    continue
                outer_value = outer_node.meta.get("val")
                if not isinstance(outer_value, torch.Tensor):
                    continue

                upper_bounds = node.args[2] if len(node.args) > 2 else None
                shape_arg = outer_node.args[0] if outer_node.args else None
                if isinstance(shape_arg, (list, tuple)):
                    for index, shape_node in enumerate(shape_arg):
                        if (
                            isinstance(shape_node, torch.fx.Node)
                            and getattr(shape_node.target, "__name__", "")
                            == "_get_symnode"
                            and shape_node.args
                            and isinstance(shape_node.args[0], str)
                            and "block_size_" in shape_node.args[0]
                            and index < len(outer_value.shape)
                        ):
                            block_id = block_id_from_key(shape_node.args[0])
                            if block_id is None:
                                continue
                            dimension = outer_value.shape[index]
                            if isinstance(dimension, torch.SymInt):
                                import sympy

                                expression = getattr(
                                    getattr(dimension, "node", None), "expr", None
                                )
                                if isinstance(expression, sympy.Symbol):
                                    context.block_symint_to_id[id(expression)] = (
                                        block_id
                                    )

                if isinstance(shape_arg, (list, tuple)):
                    concrete_shape = context.shape_from_nodes(
                        list(shape_arg), "iter_arg"
                    )
                    if isinstance(upper_bounds, (list, tuple)):
                        for index, upper_bound in enumerate(upper_bounds):
                            if index >= len(concrete_shape):
                                break
                            try:
                                upper_bound_int = int(upper_bound)
                            except Exception:
                                continue
                            if upper_bound_int > 0:
                                concrete_shape[index] = min(
                                    concrete_shape[index], upper_bound_int
                                )

                    concrete_value = torch.zeros(
                        concrete_shape, dtype=outer_value.dtype
                    )
                    placeholder.meta["val"] = concrete_value
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
                        elif target_name in ("sym_size.int", "sym_size_int"):
                            tensor_arg = body_node.args[0] if body_node.args else None
                            dimension_arg = (
                                body_node.args[1] if len(body_node.args) > 1 else None
                            )
                            if (
                                tensor_arg is placeholder
                                and isinstance(dimension_arg, int)
                                and dimension_arg < len(concrete_shape)
                            ):
                                body_node.meta["val"] = concrete_shape[dimension_arg]
                        elif target_name == "load":
                            load_index_nodes = (
                                body_node.args[1] if len(body_node.args) > 1 else None
                            )
                            if load_index_nodes is not None:
                                try:
                                    load_shape = context.shape_from_nodes(
                                        list(load_index_nodes), "load"
                                    )
                                    old_value = body_node.meta.get("val")
                                    if isinstance(old_value, torch.Tensor) and len(
                                        load_shape
                                    ) == len(old_value.shape):
                                        body_node.meta["val"] = torch.zeros(
                                            load_shape, dtype=old_value.dtype
                                        )
                                except Exception:
                                    pass
                else:
                    placeholder.meta["val"] = outer_value
                    for body_node in body_graph.nodes:
                        if (
                            body_node.op == "call_function"
                            and getattr(body_node.target, "__name__", "") == "_new_var"
                            and body_node.args
                            and body_node.args[0] is placeholder
                        ):
                            body_node.meta["val"] = outer_value
