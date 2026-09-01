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
        """Return the output name and fake tensor using stable precedence rules.

        Precedence (each step is authoritative for the cases it covers, not a
        shape/name-based guess):
        1. The device-IR ``output`` node's own tensor, matched by identity
           against a kernel parameter.
        2. The single tensor written by a ``store`` whose target isn't an
           input parameter (the common case: outputs are typically created
           fresh inside the kernel, e.g. via ``torch.empty``/``torch.zeros``).
        3. A single typed fallback: the unique non-input tensor parameter, or
           (failing that) the device-IR output tensor, or the last parameter.
        """
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
            if isinstance(output_arg, (list, tuple)):
                tensor_like_count = sum(
                    1
                    for item in output_arg
                    if isinstance(item, torch.fx.Node)
                    and isinstance(item.meta.get("val"), torch.Tensor)
                )
                if tensor_like_count > 1:
                    from .support import UnsupportedOperationError

                    raise UnsupportedOperationError(
                        "multi-output kernel",
                        reason=(
                            "The MLIR backend only supports kernels that return a "
                            "single tensor"
                        ),
                        alternatives=[
                            "Split the kernel into separate single-output kernels"
                        ],
                    )
                if output_arg:
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

        input_names = {arg.arg for arg in hf.args.args}
        input_tensor_ids = {id(tensor) for _, tensor in tensor_params}
        store_targets: list[torch.Tensor] = []
        seen_ids: set[int] = set()
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

        if len(store_targets) == 1:
            return "__store_target__", store_targets[0]
        if len(store_targets) > 1:
            # Multiple distinct non-input tensors are each written by their
            # own `store`, i.e. a genuine multi-output kernel (`return a, b`).
            # This device-IR shape is different from a plain single-output
            # kernel's: the per-loop-body FX graph's own `output` node stays
            # trivial (`None`) since outputs are mutated host tensors, not
            # graph results, so the list/tuple check above never observes
            # more than one tensor-like value there and never fires for this
            # case. Detect it here instead, where it reliably shows up as 2+
            # distinct non-input store targets, rather than silently falling
            # through to an arbitrary (and wrong) single-output guess.
            from .support import UnsupportedOperationError

            raise UnsupportedOperationError(
                "multi-output kernel",
                reason=(
                    "The MLIR backend only supports kernels that return a single tensor"
                ),
                alternatives=["Split the kernel into separate single-output kernels"],
            )

        non_input_tensors = [
            (name, tensor) for name, tensor in tensor_params if name not in input_names
        ]
        if len(non_input_tensors) == 1:
            return non_input_tensors[0]
        if output_meta_tensor is not None:
            return "__output_meta__", output_meta_tensor
        return tensor_params[-1]
