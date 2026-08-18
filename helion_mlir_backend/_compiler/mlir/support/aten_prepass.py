"""Preprocessing helpers for ATen nodes before MLIR code generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction


def refresh_aten_tensor_meta(host_function: HostFunction) -> None:
    """Refresh tensor-valued ATen metadata from current fake input shapes."""
    from ..aten_lowering import broadcast_target_shape
    from ..aten_lowering import is_aten_op
    from ..aten_lowering import normalized_aten_args

    for graph_info in host_function.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if not is_aten_op(node):
                continue

            eval_args: list[object] = []
            can_eval = True
            for arg in normalized_aten_args(node):
                if isinstance(arg, torch.fx.Node):
                    value = arg.meta.get("val")
                    if isinstance(value, torch.Tensor):
                        eval_args.append(
                            torch.zeros(
                                tuple(int(dim) for dim in value.shape),
                                dtype=value.dtype,
                            )
                        )
                    elif value is not None:
                        eval_args.append(value)
                    else:
                        can_eval = False
                        break
                else:
                    eval_args.append(arg)

            if not can_eval:
                continue

            target_name = str(node.target)
            if any(
                op in target_name
                for op in (
                    "aten.add.Tensor",
                    "aten.mul.Tensor",
                    "aten.sub.Tensor",
                    "aten.div.Tensor",
                )
            ):
                tensor_arg_idxs = [
                    index
                    for index, arg in enumerate(eval_args)
                    if isinstance(arg, torch.Tensor)
                ]
                if len(tensor_arg_idxs) >= 2:
                    target_shape = broadcast_target_shape(
                        [
                            eval_args[index]
                            for index in tensor_arg_idxs
                            if isinstance(eval_args[index], torch.Tensor)
                        ]
                    )
                    if target_shape is not None:
                        for index in tensor_arg_idxs:
                            tensor_arg = eval_args[index]
                            if isinstance(tensor_arg, torch.Tensor) and tuple(
                                int(size) for size in tensor_arg.shape
                            ) != tuple(target_shape):
                                eval_args[index] = torch.zeros(
                                    target_shape, dtype=tensor_arg.dtype
                                )

            eval_kwargs: dict[str, object] = {}
            for key, value in node.kwargs.items():
                if isinstance(value, torch.fx.Node):
                    meta_value = value.meta.get("val")
                    if isinstance(meta_value, torch.Tensor):
                        eval_kwargs[key] = torch.zeros(
                            tuple(int(dim) for dim in meta_value.shape),
                            dtype=meta_value.dtype,
                        )
                    elif meta_value is not None:
                        eval_kwargs[key] = meta_value
                    else:
                        can_eval = False
                        break
                else:
                    eval_kwargs[key] = value

            if not can_eval:
                continue

            try:
                with torch.no_grad():
                    output = node.target(*eval_args, **eval_kwargs)
            except Exception:
                continue

            tensor_output: torch.Tensor | None = None
            if isinstance(output, torch.Tensor):
                tensor_output = output
            elif isinstance(output, (list, tuple)):
                tensor_output = next(
                    (item for item in output if isinstance(item, torch.Tensor)),
                    None,
                )

            if tensor_output is not None:
                node.meta["val"] = torch.zeros(
                    tuple(int(dim) for dim in tensor_output.shape),
                    dtype=tensor_output.dtype,
                )
