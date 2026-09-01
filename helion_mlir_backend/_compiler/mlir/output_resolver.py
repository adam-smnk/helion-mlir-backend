"""Resolve the output tensor(s) represented by Helion device IR."""

from __future__ import annotations

import torch
import torch.fx


class OutputTensorResolver:
    """Select a kernel's output(s) using device-IR metadata and Helion conventions."""

    def __init__(self, host_function: object) -> None:
        self.host_function = host_function

    def resolve_all_in_graphs(
        self,
        graphs: list[torch.fx.Graph],
        exclude_tensor_ids: set[int] | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        """Every distinct tensor written by a ``store`` within *graphs*.

        Simplified relative to :meth:`resolve_all`: intended for a scoped
        subset of the kernel's graphs (e.g. one ``hl.barrier()``-separated
        phase's own root + nested loop-body graphs), where the whole-kernel
        "device-IR output node" / "unique non-input parameter" precedence
        steps don't apply. Excludes the kernel's own declared-parameter
        tensors (or, if given, *exclude_tensor_ids* instead).
        """
        hf = self.host_function
        if exclude_tensor_ids is None:
            exclude_tensor_ids = {
                id(value)
                for value in hf.params.arguments.values()
                if isinstance(value, torch.Tensor)
            }
        seen_ids: set[int] = set()
        store_targets: list[torch.Tensor] = []
        for graph in graphs:
            for node in graph.nodes:
                if (
                    node.op == "call_function"
                    and getattr(node.target, "__name__", "") == "store"
                    and len(node.args) >= 1
                    and isinstance(node.args[0], torch.fx.Node)
                ):
                    destination = node.args[0].meta.get("val")
                    if (
                        isinstance(destination, torch.Tensor)
                        and id(destination) not in exclude_tensor_ids
                        and id(destination) not in seen_ids
                    ):
                        seen_ids.add(id(destination))
                        store_targets.append(destination)
        return [("__store_target__", t) for t in store_targets]

    def resolve_all(
        self, tensor_params: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Return every distinct output name/tensor using stable precedence rules.

        Precedence (each step is authoritative for the cases it covers, not a
        shape/name-based guess):
        1. The device-IR ``output`` node's own tensor(s), matched by identity
           against a kernel parameter.
        2. Every distinct tensor written by a ``store`` whose target isn't an
           input parameter (the common case: outputs are typically created
           fresh inside the kernel, e.g. via ``torch.empty``/``torch.zeros``).
        3. A single typed fallback: the unique non-input tensor parameter, or
           (failing that) the device-IR output tensor, or the last parameter.
        """
        hf = self.host_function
        output_meta_tensor: torch.Tensor | None = None
        resolved: list[tuple[str, torch.Tensor]] = []
        seen_ids: set[int] = set()

        for graph_info in hf.device_ir.graphs:
            output_node = next(
                (node for node in graph_info.graph.nodes if node.op == "output"),
                None,
            )
            if output_node is None or not output_node.args:
                continue
            output_arg = output_node.args[0]
            candidates = (
                list(output_arg)
                if isinstance(output_arg, (list, tuple))
                else [output_arg]
            )
            for item in candidates:
                if not isinstance(item, torch.fx.Node):
                    continue
                output_value = item.meta.get("val")
                if not isinstance(output_value, torch.Tensor):
                    continue
                output_meta_tensor = output_value
                if id(output_value) in seen_ids:
                    continue
                for name, tensor in tensor_params:
                    if tensor is output_value:
                        seen_ids.add(id(output_value))
                        resolved.append((name, tensor))
                        break

        if resolved:
            return resolved

        input_names = {arg.arg for arg in hf.args.args}
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
                        and id(destination) not in seen_ids
                    ):
                        seen_ids.add(id(destination))
                        store_targets.append(destination)

        if store_targets:
            # Every distinct non-input tensor written by its own `store` is a
            # genuine output. This device-IR shape differs from a single-
            # output kernel's only in count: the per-loop-body FX graph's own
            # `output` node stays trivial (`None`) either way, since outputs
            # are mutated host tensors, not graph results -- so the count of
            # distinct store targets is the only reliable signal here.
            return [("__store_target__", t) for t in store_targets]

        non_input_tensors = [
            (name, tensor) for name, tensor in tensor_params if name not in input_names
        ]
        if len(non_input_tensors) == 1:
            return [non_input_tensors[0]]
        if output_meta_tensor is not None:
            return [("__output_meta__", output_meta_tensor)]
        return [tensor_params[-1]]
