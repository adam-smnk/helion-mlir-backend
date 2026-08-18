"""Outer parallel control-flow lowering for MLIR kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch

    from ..build_context import BuildContext


def build_kernel_body(ctx: BuildContext, out_tensor: torch.Tensor) -> ir.Value:
    """Build the outer ``scf.forall`` and its parallel insert terminator."""
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..support import torch_dtype_to_mlir

    grid_block_ids: list[int] = []
    for ids in ctx.host_function.device_ir.grid_block_ids:
        grid_block_ids.extend(ids)

    out_shape = [int(dim) for dim in out_tensor.shape]
    lbs = [0] * len(grid_block_ids)
    ubs = [out_shape[index] for index in range(len(grid_block_ids))]
    steps = [ctx.block_id_to_size[block_id] for block_id in grid_block_ids]

    for block_id, upper_bound in zip(grid_block_ids, ubs, strict=False):
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
        grid_block_ids,
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
    ub_for_match = ub_static
    # The block id from the node is authoritative; only re-derive it when that id
    # is already bound to an enclosing loop.
    if block_id in ctx.block_id_to_iv:
        candidates = [
            bid
            for bid, size in ctx.block_id_to_size.items()
            if bid not in ctx.block_id_to_iv
            and size > 0
            and ub_for_match is not None
            and ub_for_match % size == 0
        ]
        if candidates:
            largest = max(ctx.block_id_to_size[bid] for bid in candidates)
            tied = [bid for bid in candidates if ctx.block_id_to_size[bid] == largest]
            if len(tied) > 1:
                raise NodeLoweringError(
                    node,
                    reason=(
                        f"Ambiguous block id for nested loop: block ids {sorted(tied)} "
                        f"all have size {largest}"
                    ),
                    recovery_hint=(
                        "Use distinct block_sizes for nested tile dimensions; equal "
                        "sizes cannot be told apart and would silently miscompile"
                    ),
                )
            block_id = tied[0]
    step = ctx.block_id_to_size.get(block_id, ub_static if ub_static is not None else 1)
    body_graph_info = ctx.host_function.device_ir.graphs[body_graph_id]
    body_graph = body_graph_info.graph
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
                full_shape = list(target_type.shape)
                elem_ty = target_type.element_type
            else:
                if isinstance(value_meta, torch.Tensor):
                    value_shape = [int(d) for d in value_meta.shape]
                    elem_ty = torch_dtype_to_mlir(value_meta.dtype)
                else:
                    value_shape = [1 for _ in index_nodes]
                    elem_ty = torch_dtype_to_mlir(torch.float32)
                full_shape = list(value_shape)
            rank = len(full_shape)
            sym_to_block_id = ctx.build_sym_to_block_id()
            dim_block_ids: list[int | None] = []
            inner_dim: int | None = None
            for dim, idx_node in enumerate(index_nodes):
                if dim >= rank:
                    break
                dim_bid = ctx.infer_block_id_from_index(idx_node, sym_to_block_id)
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
                    dim_bid = dim_block_ids[dim]
                    if dim == inner_dim or dim_bid == block_id:
                        tile_shape.append(ub_static if ub_static is not None else step)
                        flush_offsets.append(ctx.index_const(0))
                    elif (
                        isinstance(dim_bid, int)
                        and dim_bid in active_outer_block_ids
                        and dim_bid in ctx.block_id_to_size
                    ):
                        tile_shape.append(ctx.block_id_to_size[dim_bid])
                        flush_offsets.append(ctx.block_id_to_iv[dim_bid])
                    elif (
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
        ctx.forall_insert_slices.append(
            (final_tile, synthetic_store_ctx["flush_offsets"], None)
        )
    return for_op
