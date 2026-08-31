"""Outer parallel control-flow lowering for MLIR kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch

    from ..build_context import BuildContext


def build_kernel_body(ctx: BuildContext, out_tensor: torch.Tensor) -> ir.Value:
    """Build the outer ``scf.forall`` and its parallel insert terminator.

    Maps each grid block_id to its actual destination dimension (not just positional).
    """
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..support import torch_dtype_to_mlir

    # Build mapping from block_id to output dimension. ``grid_block_ids`` groups
    # block ids by the outer ``for`` statement that produced them (e.g. a single
    # ``for tile_m, tile_n in hl.tile([m, n])`` yields one entry ``[0, 1]``), so
    # the output dimension must advance per flattened block id, not per group,
    # or every block id in a multi-dim statement collapses onto one dimension.
    block_id_to_out_dim: dict[int, int] = {}
    grid_block_ids_flat: list[int] = []
    out_shape = [int(dim) for dim in out_tensor.shape]

    out_dim = 0
    for ids in ctx.host_function.device_ir.grid_block_ids:
        for block_id in ids:
            block_id_to_out_dim[block_id] = out_dim
            grid_block_ids_flat.append(block_id)
            out_dim += 1

    # Store mapping in context for terminal store lowering.
    ctx.block_id_to_out_dim = block_id_to_out_dim

    lbs = [0] * len(grid_block_ids_flat)
    ubs = [
        out_shape[block_id_to_out_dim.get(bid, idx)]
        for idx, bid in enumerate(grid_block_ids_flat)
    ]
    steps = [ctx.block_id_to_size[block_id] for block_id in grid_block_ids_flat]

    for block_id, upper_bound in zip(grid_block_ids_flat, ubs, strict=False):
        previous = ctx.block_id_to_upper_bound.get(block_id)
        if previous is None:
            ctx.block_id_to_upper_bound[block_id] = int(upper_bound)
        else:
            ctx.block_id_to_upper_bound[block_id] = min(previous, int(upper_bound))

    output_empty = tensor_d.EmptyOp(
        out_shape,
        torch_dtype_to_mlir(out_tensor.dtype),
    ).result
    forall = scf_d.ForallOp(lbs, ubs, steps, shared_outs=[output_empty])

    for block_id, induction_variable in zip(
        grid_block_ids_flat,
        forall.induction_variables,
        strict=True,
    ):
        ctx.block_id_to_iv[block_id] = induction_variable

    with ir.InsertionPoint(forall.body):
        shared_out = next(iter(forall.inner_iter_args))
        ctx.lower_root_graphs(shared_out)
        in_parallel = scf_d.InParallelOp()
        with ir.InsertionPoint(in_parallel.block):
            for value, offsets, *rest in ctx.forall_insert_slices:
                source_type = ir.RankedTensorType(value.type)
                static_sizes = (
                    list(rest[0]) if rest and rest[0] else list(source_type.shape)
                )
                rank = len(static_sizes)
                tensor_d.ParallelInsertSliceOp(
                    value,
                    shared_out,
                    offsets,
                    [],
                    [],
                    static_offsets=[ir.ShapedType.get_dynamic_size()] * rank,
                    static_sizes=static_sizes,
                    static_strides=[1] * rank,
                )

    return forall.results[0]


def lower_nested_for_loop(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a single-dimension nested scf.for loop with optional synthetic store."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir
    import torch
    import torch.fx

    from ..support import NodeLoweringError
    from ..support import block_id_from_key
    from ..support import torch_dtype_to_mlir

    body_graph_id = node.args[0]
    block_ids = list(node.args[1])
    upper_bounds = list(node.args[2])
    iter_arg_nodes = list(node.args[3])
    assert len(block_ids) == 1 and len(upper_bounds) == 1
    block_id = block_ids[0]
    ub_src = upper_bounds[0]
    ub_static: int | None = None
    ub_val: ir.Value | None = None
    if isinstance(ub_src, int):
        ub_static = int(ub_src)
    elif isinstance(ub_src, torch.fx.Node):
        ub_val = ctx.get_value(ub_src)
        if ub_val is None:
            meta_val = ub_src.meta.get("val")
            if isinstance(meta_val, torch.Tensor) and meta_val.numel() == 1:
                try:
                    ub_static = int(meta_val.item())
                except Exception:
                    ub_static = None
            elif isinstance(meta_val, (int, float)):
                ub_static = int(meta_val)
    elif isinstance(ub_src, ir.Value):
        ub_val = ub_src
    if ub_static is None and ub_val is None:
        raise NodeLoweringError(
            node,
            reason=f"Unsupported loop upper bound type: {type(ub_src).__name__}",
            recovery_hint="Ensure loop bounds are integer constants or scalar tensor values",
        )
    if ub_val is not None:
        ub_val = ctx.cast_to_index(ub_val)
    body_graph_info = ctx.host_function.device_ir.graphs[body_graph_id]
    body_graph = body_graph_info.graph

    # Helion can reuse the enclosing grid block id on a nested loop node. The
    # body still contains the inner scalar symbol, whose origin is authoritative.
    if block_id in ctx.block_id_to_iv:
        body_block_ids = {
            info[0]
            for body_node in body_graph.nodes
            if (info := ctx.node_symbol_info(body_node)) is not None
            and info[1] in {"grid", "tile_begin", "tile_end", "tile_id"}
            and info[0] not in ctx.block_id_to_iv
        }
        body_block_ids.update(
            body_block_id
            for body_node in body_graph.nodes
            if getattr(body_node.target, "__name__", "") == "_get_symnode"
            and body_node.args
            and (body_block_id := block_id_from_key(body_node.args[0])) is not None
            and body_block_id not in ctx.block_id_to_iv
        )
        if len(body_block_ids) == 1:
            block_id = next(iter(body_block_ids))
    body_scalar_kinds = {
        info[1]
        for body_node in body_graph.nodes
        if (info := ctx.node_symbol_info(body_node)) is not None
    }
    is_grid_loop = "grid" in body_scalar_kinds
    step = ctx.block_id_to_size.get(block_id, ub_static if ub_static is not None else 1)
    output_node = next(n for n in body_graph.nodes if n.op == "output")
    out_args = output_node.args[0]
    if not isinstance(out_args, (list, tuple)):
        out_args = [out_args]
    iter_pairs = [(a, ctx.get_value(a)) for a in iter_arg_nodes]
    iter_pairs = [(a, v) for a, v in iter_pairs if v is not None]
    carried_count = len(iter_pairs)
    if 0 < len(out_args) <= len(iter_pairs):
        carried_count = len(out_args)
    invariant_pairs = iter_pairs[: len(iter_pairs) - carried_count]
    carried_pairs = iter_pairs[len(iter_pairs) - carried_count :]
    iter_init_vals = [v for _, v in carried_pairs]
    active_outer_block_ids = set(ctx.block_id_to_iv.keys())
    synthetic_store_ctx: dict[str, object] | None = None
    synthetic_iter_index: int | None = None
    store_nodes = [
        n
        for n in body_graph.nodes
        if n.op == "call_function" and getattr(n.target, "__name__", "") == "store"
    ]
    if len(store_nodes) == 1:
        store_node = store_nodes[0]
        target_node = store_node.args[0]
        index_nodes = store_node.args[1]
        value_node = store_node.args[2]
        target_val = ctx.get_value(target_node)
        target_meta = (
            target_node.meta.get("val")
            if isinstance(target_node, torch.fx.Node)
            else None
        )
        value_meta = (
            value_node.meta.get("val")
            if isinstance(value_node, torch.fx.Node)
            else None
        )
        target_type: ir.RankedTensorType | None = None
        target_rank_matches = True
        if target_val is not None:
            target_type = ir.RankedTensorType(target_val.type)
            target_rank_matches = target_type.rank == len(index_nodes)
        if isinstance(index_nodes, (list, tuple)) and target_rank_matches:
            if target_val is not None:
                assert target_type is not None
                full_shape = [1] * len(index_nodes)
                for dim, idx_node in enumerate(index_nodes):
                    if dim < len(target_type.shape) and not ctx.is_scalar_index_node(
                        idx_node
                    ):
                        full_shape[dim] = int(target_type.shape[dim])
                    elif ctx.is_scalar_index_node(idx_node):
                        full_shape[dim] = 1
                    elif dim < len(target_type.shape):
                        full_shape[dim] = int(target_type.shape[dim])
                elem_ty = target_type.element_type
            else:
                if isinstance(target_meta, torch.Tensor):
                    value_shape = [int(d) for d in target_meta.shape]
                    elem_ty = torch_dtype_to_mlir(target_meta.dtype)
                elif isinstance(value_meta, torch.Tensor):
                    value_shape = [int(d) for d in value_meta.shape]
                    elem_ty = torch_dtype_to_mlir(value_meta.dtype)
                else:
                    value_shape = [1 for _ in index_nodes]
                    elem_ty = torch_dtype_to_mlir(torch.float32)
                full_shape = [1] * len(index_nodes)
                for dim in range(len(index_nodes)):
                    if dim < len(value_shape):
                        full_shape[dim] = value_shape[dim]
                    else:
                        full_shape[dim] = 1
            rank = len(full_shape)
            sym_to_block_id = ctx.build_sym_to_block_id()
            dim_block_ids: list[int | None] = []
            inner_dim: int | None = None
            for dim, idx_node in enumerate(index_nodes):
                if dim >= rank:
                    break
                dim_bid = ctx.infer_block_id_from_index(idx_node, sym_to_block_id)
                if dim_bid is None and ctx.is_scalar_index_node(idx_node):
                    info = ctx.node_symbol_info(idx_node)
                    if info is not None:
                        dim_bid = info[0]
                dim_block_ids.append(dim_bid)
                if dim_bid == block_id:
                    inner_dim = dim
            while len(dim_block_ids) < rank:
                dim_block_ids.append(None)
            if inner_dim is None:
                inner_dim = min(rank - 1, max(0, len(index_nodes) - 1))
            if inner_dim is not None:
                tile_shape: list[int] = []
                flush_offsets: list[ir.Value] = []
                outer_bids = [bid for bid in active_outer_block_ids if bid != block_id]
                fallback_outer_bid = outer_bids[0] if outer_bids else None
                for dim, dim_size in enumerate(full_shape):
                    idx_node = index_nodes[dim] if dim < len(index_nodes) else None
                    dim_bid = dim_block_ids[dim]
                    if dim == inner_dim or dim_bid == block_id:
                        tile_shape.append(ub_static if ub_static is not None else step)
                        flush_offsets.append(ctx.index_const(0))
                        continue
                    if idx_node is not None and ctx.is_scalar_index_node(idx_node):
                        tile_shape.append(1)
                        scalar_value = (
                            ctx.block_id_to_iv.get(dim_bid)
                            if isinstance(dim_bid, int)
                            else None
                        )
                        if scalar_value is None:
                            scalar_value = ctx.get_value(idx_node)
                        flush_offsets.append(
                            scalar_value
                            if scalar_value is not None
                            else ctx.index_const(0)
                        )
                        continue
                    if (
                        isinstance(dim_bid, int)
                        and dim_bid in active_outer_block_ids
                        and dim_bid in ctx.block_id_to_size
                    ):
                        tile_shape.append(ctx.block_id_to_size[dim_bid])
                        flush_offsets.append(ctx.block_id_to_iv[dim_bid])
                    elif not is_grid_loop and (
                        fallback_outer_bid is not None
                        and fallback_outer_bid in ctx.block_id_to_size
                    ):
                        tile_shape.append(ctx.block_id_to_size[fallback_outer_bid])
                        flush_offsets.append(ctx.block_id_to_iv[fallback_outer_bid])
                    else:
                        tile_shape.append(int(dim_size))
                        flush_offsets.append(ctx.index_const(0))
                tile_empty = tensor_d.EmptyOp(tile_shape, elem_ty).result
                if isinstance(elem_ty, ir.FloatType):
                    zero_attr = ir.FloatAttr.get(elem_ty, 0.0)
                else:
                    zero_attr = ir.IntegerAttr.get(elem_ty, 0)
                zero = arith_d.ConstantOp(elem_ty, zero_attr).result
                tile_init = linalg_d.fill(zero, outs=[tile_empty])
                synthetic_iter_index = len(iter_init_vals)
                iter_init_vals.append(tile_init)
                synthetic_store_ctx = {
                    "block_id": block_id,
                    "inner_dim": inner_dim,
                    "rank": rank,
                    "flush_offsets": flush_offsets,
                    "current": None,
                }
    lb_val = ctx.index_const(0)
    ub_val = (
        ub_val
        if ub_val is not None
        else ctx.index_const(ub_static if ub_static is not None else step)
    )
    step_val = ctx.index_const(step)
    for_op = scf_d.ForOp(lb_val, ub_val, step_val, iter_args=iter_init_vals)
    body_block = for_op.body
    with (
        ir.InsertionPoint(body_block),
        ctx.enter_for_loop(block_id, body_block.arguments[0]),
    ):
        placeholders = [n for n in body_graph.nodes if n.op == "placeholder"]
        if len(placeholders) > len(iter_pairs):
            placeholders = placeholders[-len(iter_pairs) :]
        invariant_placeholders = placeholders[: len(invariant_pairs)]
        for ph_node, (_, inv_val) in zip(
            invariant_placeholders, invariant_pairs, strict=False
        ):
            ctx.set_value(ph_node, inv_val)
        carried_placeholders = placeholders[len(invariant_pairs) :]
        for ph_node, body_arg in zip(
            carried_placeholders, body_block.arguments[1:], strict=False
        ):
            ctx.set_value(ph_node, body_arg)
        if synthetic_store_ctx is not None and synthetic_iter_index is not None:
            synthetic_store_ctx["current"] = body_block.arguments[
                1 + synthetic_iter_index
            ]
            with ctx.push_store_ctx(synthetic_store_ctx):
                ctx.lower_graph(body_graph)
        else:
            ctx.lower_graph(body_graph)
        yield_vals = []
        for a in out_args:
            v = ctx.get_value(a) if isinstance(a, torch.fx.Node) else None
            if v is not None:
                yield_vals.append(v)
        if synthetic_store_ctx is not None:
            current = synthetic_store_ctx.get("current")
            if current is not None:
                insert_at = (
                    synthetic_iter_index
                    if synthetic_iter_index is not None
                    else len(yield_vals)
                )
                if insert_at <= len(yield_vals):
                    yield_vals.insert(insert_at, current)
                else:
                    yield_vals.append(current)
        if len(yield_vals) != len(iter_init_vals):
            if len(yield_vals) > len(iter_init_vals):
                raise NodeLoweringError(
                    node,
                    reason=f"Loop body yielded more values than iter_args: {len(yield_vals)} > {len(iter_init_vals)}",
                    recovery_hint="Ensure loop-carried values match loop iter_args",
                )
            passthrough_count = len(iter_init_vals) - len(yield_vals)
            passthrough_vals = list(body_block.arguments[1 : 1 + passthrough_count])
            yield_vals = yield_vals + passthrough_vals
        scf_d.YieldOp(yield_vals)
    if synthetic_store_ctx is not None and synthetic_iter_index is not None:
        final_tile = for_op.results[synthetic_iter_index]
        if ctx.for_store_ctx_stack:
            parent_ctx = ctx.for_store_ctx_stack[-1]
            parent_current = parent_ctx.get("current")
            if parent_current is not None:
                parent_type = ir.RankedTensorType(parent_current.type)
                tile_type = ir.RankedTensorType(final_tile.type)
                offsets = list(synthetic_store_ctx["flush_offsets"])
                if len(offsets) != parent_type.rank:
                    offsets = offsets[: parent_type.rank]
                    offsets.extend(
                        ctx.index_const(0)
                        for _ in range(parent_type.rank - len(offsets))
                    )
                updated = tensor_d.InsertSliceOp(
                    final_tile,
                    parent_current,
                    offsets,
                    [],
                    [],
                    static_offsets=[ir.ShapedType.get_dynamic_size()]
                    * parent_type.rank,
                    static_sizes=[int(dim) for dim in tile_type.shape],
                    static_strides=[1] * parent_type.rank,
                ).result
                parent_ctx["current"] = updated
        else:
            ctx.forall_insert_slices.append(
                (final_tile, synthetic_store_ctx["flush_offsets"], None)
            )
    return for_op
