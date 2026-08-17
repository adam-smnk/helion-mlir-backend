"""Resolve the output tensor represented by Helion device IR."""

from __future__ import annotations

import torch
import torch.fx


class OutputTensorResolver:
    """Select a kernel output using device-IR metadata and Helion conventions."""

    def __init__(self, host_function: object) -> None:
        self.host_function = host_function

    def resolve(
        self, tensor_params: list[tuple[str, torch.Tensor]]
    ) -> tuple[str, torch.Tensor]:
        """Return the output name and fake tensor using stable precedence rules."""
        hf = self.host_function
        output_meta_tensor: torch.Tensor | None = None

        for graph_info in hf.device_ir.graphs:
            output_node = next(
                (node for node in graph_info.graph.nodes if node.op == "output"),
                None,
            )
            if output_node is None or not output_node.args:
                continue
            output_arg = output_node.args[0]
            if isinstance(output_arg, (list, tuple)) and output_arg:
                output_arg = output_arg[0]
            if not isinstance(output_arg, torch.fx.Node):
                continue
            output_value = output_arg.meta.get("val")
            if not isinstance(output_value, torch.Tensor):
                continue
            output_meta_tensor = output_value
            for name, tensor in tensor_params:
                if tensor is output_value:
                    return name, tensor

        if output_meta_tensor is not None:
            same_meta = [
                (name, tensor)
                for name, tensor in tensor_params
                if tuple(tensor.shape) == tuple(output_meta_tensor.shape)
                and tensor.dtype == output_meta_tensor.dtype
            ]
            if len(same_meta) == 1:
                return same_meta[0]

        input_names = {arg.arg for arg in hf.args.args}
        for name, tensor in tensor_params:
            if name == "out" and name not in input_names:
                return name, tensor

        input_tensor_ids = {id(tensor) for _, tensor in tensor_params}
        store_targets: list[torch.Tensor] = []
        for graph_info in hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if (
                    node.op == "call_function"
                    and getattr(node.target, "__name__", "") == "store"
                    and len(node.args) >= 1
                    and isinstance(node.args[0], torch.fx.Node)
                ):
                    destination = node.args[0].meta.get("val")
                    if (
                        isinstance(destination, torch.Tensor)
                        and id(destination) not in input_tensor_ids
                    ):
                        store_targets.append(destination)

        if store_targets:
            unique_targets: list[torch.Tensor] = []
            seen_ids: set[int] = set()
            for tensor in store_targets:
                if id(tensor) in seen_ids:
                    continue
                seen_ids.add(id(tensor))
                unique_targets.append(tensor)

            if len(unique_targets) == 1:
                return "__store_target__", unique_targets[0]

            if output_meta_tensor is not None:
                matching_targets = [
                    tensor
                    for tensor in unique_targets
                    if tuple(tensor.shape) == tuple(output_meta_tensor.shape)
                    and tensor.dtype == output_meta_tensor.dtype
                ]
                if len(matching_targets) == 1:
                    return "__store_target__", matching_targets[0]

            ranked_targets = sorted(
                unique_targets,
                key=lambda tensor: (len(tensor.shape), tensor.numel()),
                reverse=True,
            )
            if ranked_targets:
                return "__store_target__", ranked_targets[0]

        non_input_tensors = [
            (name, tensor) for name, tensor in tensor_params if name not in input_names
        ]
        if len(non_input_tensors) == 1:
            return non_input_tensors[0]

        if output_meta_tensor is not None:
            same_non_input_meta = [
                (name, tensor)
                for name, tensor in non_input_tensors
                if tuple(tensor.shape) == tuple(output_meta_tensor.shape)
                and tensor.dtype == output_meta_tensor.dtype
            ]
            if len(same_non_input_meta) == 1:
                return same_non_input_meta[0]

        for graph_info in hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if (
                    node.op == "call_function"
                    and getattr(node.target, "__name__", "") == "_host_tensor"
                ):
                    name = node.args[0]
                    if name not in input_names:
                        value = node.meta.get("val")
                        if isinstance(value, torch.Tensor):
                            return name, value

        if non_input_tensors:
            ranked = sorted(
                non_input_tensors,
                key=lambda item: (len(item[1].shape), item[1].numel()),
                reverse=True,
            )
            return ranked[0]

        if output_meta_tensor is not None:
            return "__output_meta__", output_meta_tensor

        return tensor_params[-1]
