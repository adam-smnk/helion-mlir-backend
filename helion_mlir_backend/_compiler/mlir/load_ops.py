"""Specialized tensor load lowering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir


def lower_flat_gather(
    builder: object,
    tensor_node: object,
    tensor_value: ir.Value,
    index_value: ir.Value,
    index_type: ir.RankedTensorType,
    tensor_type: ir.RankedTensorType,
) -> ir.Value | None:
    """Lower a flattened one-dimensional source indexed by an N-D tensor."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    gather_shape = [int(dim) for dim in index_type.shape]
    if isinstance(tensor_node, torch.fx.Node) and gather_shape:
        trailing_extent: int | None = None
        source_target_name = str(getattr(tensor_node, "target", ""))
        source_target = getattr(tensor_node.target, "__name__", "")
        is_view = tensor_node.op in ("call_function", "call_method") and (
            "aten.view" in source_target_name
            or source_target in ("view", "view.default")
        )
        if is_view and tensor_node.args:
            base_node = tensor_node.args[0]
            base_meta = (
                base_node.meta.get("val")
                if isinstance(base_node, torch.fx.Node)
                else None
            )
            if isinstance(base_meta, torch.Tensor) and base_meta.ndim >= 2:
                trailing_extent = int(base_meta.shape[-1])

        if (
            trailing_extent is None
            and source_target == "_host_tensor"
            and tensor_node.args
        ):
            alias_name = tensor_node.args[0]
            if isinstance(alias_name, str) and alias_name.endswith("_flat"):
                base_name = alias_name[: -len("_flat")]
                alias_value = builder.hf.params.arguments.get(alias_name)
                base_value = builder.hf.params.arguments.get(base_name)
                if (
                    isinstance(alias_value, torch.Tensor)
                    and isinstance(base_value, torch.Tensor)
                    and base_value.ndim >= 2
                    and alias_value.numel() == base_value.numel()
                ):
                    trailing_extent = int(base_value.shape[-1])

        if trailing_extent is not None and trailing_extent > 0:
            gather_shape[-1] = min(gather_shape[-1], trailing_extent)

    result_type = ir.RankedTensorType.get(gather_shape, tensor_type.element_type)
    generate = tensor_d.GenerateOp(result_type, [])
    body = generate.operation.regions[0].blocks.append(
        *([ir.IndexType.get()] * len(gather_shape))
    )
    with ir.InsertionPoint(body):
        indices = list(body.arguments)
        extracted_index = tensor_d.ExtractOp(
            index_value,
            indices,
            results=[index_type.element_type],
        ).result
        extracted_index = builder._cast_to_index(extracted_index)
        gathered = tensor_d.ExtractOp(
            tensor_value,
            [extracted_index],
            results=[tensor_type.element_type],
        ).result
        tensor_d.YieldOp(gathered)
    return generate.result
