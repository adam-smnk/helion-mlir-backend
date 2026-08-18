"""On-demand rebuilding of ATen helpers for concrete call-site shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx

from ..support import mlir_dtype_to_torch

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def rebuild_aten_helper_for_call(
    ctx: BuildContext,
    node: torch.fx.Node,
    input_mlir_values: list[ir.Value],
) -> tuple[str, list[ir.Type]] | None:
    """Build a helper variant using the current operand MLIR types."""
    import mlir.ir as ir

    from ..aten_lowering import collect_tensor_input_positions
    from ..aten_lowering import normalized_aten_args
    from ..aten_lowering import preprocess_aten_nodes

    if ctx.mlir_module is None:
        return None

    normalized_args = normalized_aten_args(node)
    tensor_positions = collect_tensor_input_positions(node)
    if len(tensor_positions) != len(input_mlir_values):
        return None

    backups: list[tuple[torch.fx.Node, object, object]] = []
    overrides: dict[int, torch.Tensor] = {}
    try:
        for argument_index, mlir_value in zip(
            tensor_positions, input_mlir_values, strict=True
        ):
            argument_node = normalized_args[argument_index]
            if not isinstance(argument_node, torch.fx.Node):
                continue

            ranked_type = ir.RankedTensorType(mlir_value.type)
            shape = [int(dimension) for dimension in ranked_type.shape]
            element_key = str(ranked_type.element_type)
            dtype = mlir_dtype_to_torch(
                element_key,
                default=torch.int64 if element_key == "index" else torch.float32,
            )
            if element_key not in {
                "f32",
                "f64",
                "f16",
                "bf16",
                "i1",
                "i8",
                "i16",
                "i32",
                "i64",
                "index",
            }:
                return None

            old_value = argument_node.meta.get("val")
            old_tensor_meta = argument_node.meta.get("tensor_meta")
            backups.append((argument_node, old_value, old_tensor_meta))
            concrete = torch.zeros(shape, dtype=dtype)
            argument_node.meta["val"] = concrete
            overrides[argument_index] = concrete

        try:
            rebuilt_map = preprocess_aten_nodes(
                [node],
                ctx.mlir_module,
                ctx.block_id_to_size,
                ctx.block_hint_to_id,
                ctx.block_symint_to_id,
                ctx.block_id_to_upper_bound,
                {id(node): overrides},
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "On-demand helper rebuild failed for node %s (%s): %s",
                node.name,
                node.target,
                exc,
            )
            return None

        rebuilt = rebuilt_map.get(id(node))
        if rebuilt is not None:
            ctx.node_to_aten_func[id(node)] = rebuilt
        return rebuilt
    finally:
        for argument_node, old_value, old_tensor_meta in backups:
            if old_value is None:
                argument_node.meta.pop("val", None)
            else:
                argument_node.meta["val"] = old_value
            if old_tensor_meta is None:
                argument_node.meta.pop("tensor_meta", None)
            else:
                argument_node.meta["tensor_meta"] = old_tensor_meta
