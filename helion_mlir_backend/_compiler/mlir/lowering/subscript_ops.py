"""Tensor subscript and gather lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_subscript(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower tensor-valued indexing and full-slice subscripts."""
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    if len(node.args) < 2:
        return None
    source_value = ctx.get_value(node.args[0])
    if source_value is None:
        return None

    index_candidates: list[object] = []
    for arg in node.args[1:]:
        index_candidates.extend(arg if isinstance(arg, (list, tuple)) else [arg])
    index_value = next(
        (
            ctx.get_value(candidate)
            for candidate in index_candidates
            if ctx.get_value(candidate) is not None
        ),
        None,
    )
    try:
        source_type = ir.RankedTensorType(source_value.type)
    except Exception:
        return None
    element_type = source_type.element_type

    def is_full_slice(spec: object) -> bool:
        return (
            isinstance(spec, slice)
            and spec.start is None
            and spec.stop is None
            and spec.step is None
        )

    if (
        index_value is not None
        and source_type.rank == 1
        and any(
            not isinstance(spec, (None.__class__, slice)) for spec in index_candidates
        )
    ):
        index_type = ir.RankedTensorType(index_value.type)
        result_shape = [int(dim) for dim in index_type.shape]
        if not result_shape:
            index = index_value
            if not isinstance(index.type, ir.IndexType):
                index = tensor_d.ExtractOp(
                    index, [], results=[ir.IndexType.get()]
                ).result
            return tensor_d.ExtractOp(
                source_value, [index], results=[element_type]
            ).result

        result_type = ir.RankedTensorType.get(result_shape, element_type)
        generate = tensor_d.GenerateOp(result_type, [])
        body = generate.operation.regions[0].blocks.append(
            *([ir.IndexType.get()] * len(result_shape))
        )
        with ir.InsertionPoint(body):
            indices = list(body.arguments)
            extracted_index = tensor_d.ExtractOp(
                index_value, indices, results=[ir.IndexType.get()]
            ).result
            if not isinstance(extracted_index.type, ir.IndexType):
                extracted_index = ctx.cast_to_index(extracted_index)
            gathered = tensor_d.ExtractOp(
                source_value, [extracted_index], results=[element_type]
            ).result
            tensor_d.YieldOp(gathered)
        return generate.result

    scalar_index_dims: list[int] = []
    for dimension, spec in enumerate(index_candidates[: source_type.rank]):
        if isinstance(spec, torch.fx.Node) and ctx.is_scalar_index_node(spec):
            scalar_index_dims.append(dimension)

    if scalar_index_dims:
        offsets: list[ir.Value] = []
        sizes: list[int] = []
        for dimension in range(source_type.rank):
            spec = (
                index_candidates[dimension]
                if dimension < len(index_candidates)
                else slice(None)
            )
            if spec is None:
                offsets.append(ctx.index_const(0))
                sizes.append(int(source_type.shape[dimension]))
                continue
            if (
                isinstance(spec, slice)
                and spec.start is None
                and spec.stop is None
                and spec.step is None
            ):
                offsets.append(ctx.index_const(0))
                sizes.append(int(source_type.shape[dimension]))
                continue
            if isinstance(spec, torch.fx.Node) and ctx.is_scalar_index_node(spec):
                scalar_offset = ctx.get_value(spec)
                offsets.append(
                    ctx.cast_to_index(scalar_offset)
                    if scalar_offset is not None
                    else ctx.index_const(0)
                )
                sizes.append(1)
                continue
            return None

        result_shape = [
            extent
            for dimension, extent in enumerate(sizes)
            if dimension not in scalar_index_dims
        ]
        result_type = ir.RankedTensorType.get(result_shape, element_type)
        return tensor_d.ExtractSliceOp(
            result_type,
            source_value,
            offsets,
            [],
            [],
            static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
            static_sizes=sizes,
            static_strides=[1] * len(offsets),
        ).result

    if any(isinstance(spec, (int, float)) for spec in index_candidates):
        return None
    result_shape = ctx.shape_from_node_meta(node)
    if result_shape is None:
        result_shape = [int(dim) for dim in source_type.shape]
        result_shape.extend(1 for spec in index_candidates if spec is None)
    if not result_shape:
        return None

    result_type = ir.RankedTensorType.get(result_shape, element_type)
    generate = tensor_d.GenerateOp(result_type, [])
    body = generate.operation.regions[0].blocks.append(
        *([ir.IndexType.get()] * len(result_shape))
    )
    source_indices: list[ir.Value] = []
    output_dim = 0
    source_dim = 0
    with ir.InsertionPoint(body):
        indices = list(body.arguments)
        for spec in index_candidates:
            if spec is None:
                continue
            if not is_full_slice(spec):
                return None
            if source_dim >= source_type.rank or output_dim >= len(indices):
                return None
            source_indices.append(indices[output_dim])
            source_dim += 1
            output_dim += 1
        if source_dim != source_type.rank:
            return None
        gathered = tensor_d.ExtractOp(
            source_value, source_indices, results=[element_type]
        ).result
        tensor_d.YieldOp(gathered)
    return generate.result
