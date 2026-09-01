"""Specialized tensor load lowering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_flat_gather(
    ctx: BuildContext,
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

        if trailing_extent is None and source_target == "_host_tensor":
            alias_value = tensor_node.meta.get("val")
            if isinstance(alias_value, torch.Tensor):
                # A host-side flattened view (e.g. ``x_flat = x_data.view(-1)``
                # written outside the tiled loop) shares storage with its
                # source tensor; find that source among the kernel's own
                # parameters by storage identity rather than guessing from
                # the captured variable's name.
                try:
                    alias_storage = alias_value.untyped_storage()
                except Exception:
                    # torch.Tensor.untyped_storage() can raise for unusual
                    # tensor backends; treat as "no storage identity available".
                    alias_storage = None
                if alias_storage is not None:
                    for candidate in ctx.host_function.params.arguments.values():
                        if (
                            isinstance(candidate, torch.Tensor)
                            and candidate.ndim >= 2
                            and candidate is not alias_value
                        ):
                            try:
                                same_storage = (
                                    candidate.untyped_storage() is alias_storage
                                )
                            except Exception:
                                # Same rationale as above: unusual tensor
                                # backends can raise here.
                                same_storage = False
                            if same_storage:
                                trailing_extent = int(candidate.shape[-1])
                                break

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
        extracted_index = ctx.cast_to_index(extracted_index)
        gathered = tensor_d.ExtractOp(
            tensor_value,
            [extracted_index],
            results=[tensor_type.element_type],
        ).result
        tensor_d.YieldOp(gathered)
    return generate.result
