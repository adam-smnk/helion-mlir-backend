"""Ordinary Helion tile loads lowered to tensor.extract_slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_load(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a Helion load to a static tensor extract slice."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    tensor_node = node.args[0]
    index_nodes = node.args[1]
    tensor_value = ctx.get_value(tensor_node)
    assert tensor_value is not None, f"No value for tensor node {tensor_node}"
    tensor_type = ir.RankedTensorType(tensor_value.type)
    ndim = len(tensor_type.shape)

    if ndim == 1 and len(index_nodes) == 1:
        gather_index_value = ctx.get_value(index_nodes[0])
        if gather_index_value is not None:
            try:
                gather_index_type = ir.RankedTensorType(gather_index_value.type)
            except Exception:
                gather_index_type = None
            if gather_index_type is not None and gather_index_type.rank >= 1:
                from .load_ops import lower_flat_gather

                gathered = lower_flat_gather(
                    ctx,
                    tensor_node,
                    tensor_value,
                    gather_index_value,
                    gather_index_type,
                    tensor_type,
                )
                if gathered is not None:
                    return gathered

    offsets: list[ir.Value] = []
    sizes: list[int] = []
    sym_to_block_id = ctx.build_sym_to_block_id()
    used_block_ids: set[int] = set()
    forced: dict[int, int] = {}

    # `hl.grid` indices and `tile.begin`-style positions select a single element
    # along their dimension, which is then dropped from the loaded tile.
    scalar_dims = {
        dimension
        for dimension, index_node in enumerate(index_nodes)
        if dimension < ndim and ctx.is_scalar_index_node(index_node)
    }

    if ctx.for_store_ctx_stack:
        store_context = ctx.for_store_ctx_stack[-1]
        inner_block_id = int(store_context.get("block_id", -1))
        inner_dimension = int(store_context.get("inner_dim", -1))
        rank = int(store_context.get("rank", ndim))
        if 0 <= inner_dimension < rank:
            forced[inner_dimension] = inner_block_id
            outer_candidates = [
                block_id
                for block_id in ctx.block_id_to_iv
                if block_id != inner_block_id
            ]
            if len(outer_candidates) == 1:
                for dimension in range(rank):
                    if dimension != inner_dimension:
                        forced[dimension] = outer_candidates[0]

    result_value = node.meta.get("val")
    result_sizes: list[int] | None = None
    if isinstance(result_value, torch.Tensor):
        result_sizes = [int(dim) for dim in result_value.shape]
        if len(result_sizes) < ndim:
            scalar_count = ndim - len(result_sizes)
            scalar_dims = set(sorted(scalar_dims)[:scalar_count])

    for dimension, index_node in enumerate(index_nodes):
        if dimension >= ndim:
            break

        if dimension in scalar_dims:
            scalar_offset = ctx.get_value(index_node)
            offsets.append(
                ctx.cast_to_index(scalar_offset)
                if scalar_offset is not None
                else ctx.index_const(0)
            )
            sizes.append(1)
            continue

        if isinstance(index_node, slice):
            offsets.append(ctx.index_const(0))
            extent = int(tensor_type.shape[dimension])
            if result_sizes is not None:
                result_index = dimension - sum(1 for d in scalar_dims if d < dimension)
                if result_index < len(result_sizes):
                    extent = min(extent, int(result_sizes[result_index]))
            sizes.append(extent)
            continue

        # Result metadata omits scalar-indexed dimensions, so realign the index.
        result_index = dimension - sum(1 for d in scalar_dims if d < dimension)
        index_extent: int | None = None
        if isinstance(index_node, torch.fx.Node):
            index_value = ctx.get_value(index_node)
            if index_value is not None:
                try:
                    index_type = ir.RankedTensorType(index_value.type)
                    if index_type.rank == 1:
                        index_extent = int(index_type.shape[0])
                except Exception:
                    pass
        is_forced = dimension in forced
        block_id, index_bias = ctx.infer_index_block_and_bias(
            index_node, sym_to_block_id
        )
        if is_forced:
            block_id = forced[dimension]
            index_bias = 0
        allow_fallback = isinstance(index_node, torch.fx.Node)
        if (
            allow_fallback
            and block_id is None
            and result_sizes is not None
            and result_index < len(result_sizes)
        ):
            matching = [
                candidate
                for candidate in ctx.block_id_to_iv
                if ctx.block_id_to_size.get(candidate) == result_sizes[result_index]
                and candidate not in used_block_ids
            ]
            if len(matching) == 1:
                block_id = matching[0]
        if allow_fallback and block_id is None:
            extent = int(tensor_type.shape[dimension])
            matching = [
                candidate
                for candidate in ctx.block_id_to_iv
                if candidate not in used_block_ids
                and ctx.block_id_to_upper_bound.get(candidate) == extent
            ]
            if len(matching) == 1:
                block_id = matching[0]

        if block_id is not None and block_id in ctx.block_id_to_iv:
            offset = ctx.block_id_to_iv[block_id]
            if index_bias:
                offset = arith_d.AddIOp(offset, ctx.index_const(index_bias)).result
            offsets.append(offset)
            used_block_ids.add(block_id)
        else:
            offsets.append(ctx.index_const(0))

        extent = int(tensor_type.shape[dimension])
        if block_id is not None and block_id in ctx.block_id_to_size:
            configured = ctx.block_id_to_size[block_id]
            upper_bound = ctx.block_id_to_upper_bound.get(block_id)
            if upper_bound is not None and upper_bound > 0:
                configured = min(configured, upper_bound)
            if (
                not is_forced
                and result_sizes is not None
                and result_index < len(result_sizes)
            ):
                sizes.append(min(configured, result_sizes[result_index], extent))
            else:
                sizes.append(min(configured, extent))
        elif index_extent is not None:
            sizes.append(min(index_extent, extent))
        elif result_sizes is not None and result_index < len(result_sizes):
            sizes.append(min(result_sizes[result_index], extent))
        else:
            sizes.append(extent)

    if scalar_dims:
        expanded_sizes: list[int] = [1] * ndim
        result_index = 0
        for dimension in range(ndim):
            if dimension in scalar_dims:
                expanded_sizes[dimension] = 1
            elif result_index < len(sizes):
                expanded_sizes[dimension] = sizes[result_index]
                result_index += 1
            else:
                expanded_sizes[dimension] = int(tensor_type.shape[dimension])
        sizes = expanded_sizes
        if result_sizes is not None and len(result_sizes) < ndim:
            sizes = [1] * (ndim - len(result_sizes)) + result_sizes

    result_shape = [
        extent for dimension, extent in enumerate(sizes) if dimension not in scalar_dims
    ]
    if result_sizes is not None and len(result_sizes) < ndim:
        result_shape = result_sizes
    if not result_shape:
        result_shape = [1]
    result_type = ir.RankedTensorType.get(result_shape, tensor_type.element_type)
    return tensor_d.ExtractSliceOp(
        result_type,
        tensor_value,
        offsets,
        [],
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
        static_sizes=sizes,
        static_strides=[1] * len(offsets),
    ).result
