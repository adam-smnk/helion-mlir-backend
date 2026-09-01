"""Outer parallel control-flow lowering for MLIR kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch

    from ..build_context import BuildContext
    from .for_store_context import ForStoreContext


def _block_id_to_out_dim_from_terminal_store(
    ctx: BuildContext, grid_block_ids: list[int]
) -> dict[int, int] | None:
    """Find the store that writes the final output and map each of its index
    positions to the block id it resolves to.

    This is authoritative: a store's index position *is* the output
    dimension, regardless of the declaration order of the enclosing loops
    (e.g. ``out[tm, panel, :] = ...`` writes ``panel``'s block id to output
    dimension 1 even though the ``panel`` loop is declared before ``tm``'s).

    A store's index list may also resolve block ids for loops that are *not*
    part of the outer parallel grid (e.g. a nested ``hl.tile()`` loop), so a
    store is accepted once every outer grid block id is found among its
    resolved indices -- a stronger, less accidental signal than matching on
    tensor shape alone (multiple tensors can share a shape) -- while any
    other resolved block ids are simply ignored. Returns ``None`` if no such
    store is found.
    """
    from ..support.index_meta import resolve_index_descriptor

    expected_block_ids = set(grid_block_ids)

    for graph_info in ctx.host_function.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "store":
                continue
            index_nodes = node.args[1]
            if not isinstance(index_nodes, (list, tuple)):
                continue

            mapping: dict[int, int] = {}
            for dim, index_node in enumerate(index_nodes):
                if isinstance(index_node, slice):
                    continue
                descriptor = resolve_index_descriptor(ctx, index_node)
                if descriptor.block_id is not None:
                    mapping[descriptor.block_id] = dim
            if expected_block_ids and expected_block_ids.issubset(mapping):
                return {bid: mapping[bid] for bid in expected_block_ids}
    return None


def build_kernel_body(ctx: BuildContext, out_tensor: torch.Tensor) -> ir.Value:
    """Build the outer ``scf.forall`` and its parallel insert terminator.

    Maps each grid block_id to its actual destination dimension (not just positional).
    """
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..support import torch_dtype_to_mlir

    out_shape = [int(dim) for dim in out_tensor.shape]

    # ``grid_block_ids`` groups block ids by the outer ``for`` statement that
    # produced them (e.g. a single ``for tile_m, tile_n in hl.tile([m, n])``
    # yields one entry ``[0, 1]``), so the flattened list must advance per
    # block id, not per group, or every block id in a multi-dim statement
    # collapses onto one dimension.
    grid_block_ids_flat: list[int] = []
    for ids in ctx.host_function.device_ir.grid_block_ids:
        grid_block_ids_flat.extend(ids)

    # Prefer the authoritative mapping derived from the terminal store's own
    # index expression: loop declaration order does not necessarily match the
    # order block ids are indexed in the output (e.g. ``out[tm, panel, :]``
    # with ``panel``'s loop declared before ``tm``'s). Fall back to loop
    # declaration order only if no matching terminal store is found.
    block_id_to_out_dim = _block_id_to_out_dim_from_terminal_store(
        ctx, grid_block_ids_flat
    )
    if block_id_to_out_dim is None:
        block_id_to_out_dim = {
            block_id: out_dim for out_dim, block_id in enumerate(grid_block_ids_flat)
        }

    lbs = [0] * len(grid_block_ids_flat)
    ubs = [
        out_shape[block_id_to_out_dim.get(bid, idx)]
        for idx, bid in enumerate(grid_block_ids_flat)
    ]
    steps = [ctx.block_id_to_size[block_id] for block_id in grid_block_ids_flat]

    # The outer scf.forall emits one statically-sized extract/insert per
    # iteration (no per-iteration dynamic clamp for a ragged last tile), so a
    # dimension that's part of a COMBINED multi-dim tile (e.g. hl.tile([m, n]))
    # and needs more than one iteration must divide evenly by its block size;
    # a ragged last iteration would read/write past the tensor's real bound.
    # A single-dimension hl.tile() is unaffected: its own tile.end/mask-based
    # dynamic clamping (see tile_index_ops.scalar_tile_value) already handles
    # raggedness correctly. A single iteration (step >= extent) is also
    # unaffected since slice_plan already clamps that case statically.
    combined_block_ids = {
        bid
        for ids in ctx.host_function.device_ir.grid_block_ids
        if len(ids) > 1
        for bid in ids
    }
    for block_id, step, ub in zip(grid_block_ids_flat, steps, ubs, strict=True):
        if block_id in combined_block_ids and step < ub and ub % step != 0:
            from ..support import UnsupportedOperationError

            raise UnsupportedOperationError(
                "ragged combined-tile block size",
                reason=(
                    f"block_id {block_id}: dimension of size {ub} is not evenly "
                    f"divisible by block size {step}, and needs more than one "
                    "iteration; this backend does not yet support a "
                    "dynamically-sized boundary tile in this position"
                ),
                alternatives=[
                    "choose a block size that evenly divides this dimension",
                    "restructure the kernel so this dimension needs only one iteration",
                ],
            )

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
        # Bound for the whole compile (no enclosing scope to restore to),
        # unlike nested scf.for levels which use ctx.enter_for_loop's
        # save/restore.
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


def _find_reused_block_id(
    ctx: BuildContext, graph: torch.fx.Graph, max_depth: int = 8
) -> int | None:
    """Resolve the single new block id introduced by a loop whose ``_for_loop``
    node reused an already-mapped (outer) block id.

    Helion can nest several loop levels between the reused id's introduction
    and the level whose real identity we need (e.g. ``grid -> grid -> tile``
    3+ levels deep), so the body at this exact level may be a pure wrapper —
    nothing but one further ``_for_loop`` call. Unwrap those wrapper levels
    one at a time until a body with actual scalar symbol references is found.
    Returns ``None`` if no single unambiguous candidate is found.
    """
    device_ir = ctx.host_function.device_ir
    current_graph = graph
    for _ in range(max_depth):
        candidates = {
            info[0]
            for body_node in current_graph.nodes
            if (info := ctx.node_symbol_info(body_node)) is not None
            and info[1] in {"grid", "tile_begin", "tile_end", "tile_id"}
            and info[0] not in ctx.block_id_to_iv
        }
        if candidates:
            return next(iter(candidates)) if len(candidates) == 1 else None
        call_nodes = [n for n in current_graph.nodes if n.op == "call_function"]
        if (
            len(call_nodes) == 1
            and getattr(call_nodes[0].target, "__name__", "") == "_for_loop"
        ):
            current_graph = device_ir.graphs[call_nodes[0].args[0]].graph
            continue
        break
    return None


def _find_descendant_store(
    ctx: BuildContext, graph: torch.fx.Graph, max_depth: int = 16
) -> torch.fx.Node | None:
    """DFS through nested ``_for_loop`` bodies for the first ``store`` call.

    Scoped to true descendants of ``graph`` only (unlike a global scan over
    every graph), so it is safe to use for detecting whether an intermediate
    loop level with no store of its own (a pure pass-through, e.g. the
    middle loop of ``grid -> grid -> tile``) must thread an accumulator down
    to a deeper level that does have one. Works to arbitrary nesting depth.
    """
    device_ir = ctx.host_function.device_ir
    stack: list[tuple[torch.fx.Graph, int]] = [(graph, 0)]
    while stack:
        current_graph, depth = stack.pop()
        if depth > max_depth:
            continue
        for graph_node in current_graph.nodes:
            if (
                graph_node.op == "call_function"
                and getattr(graph_node.target, "__name__", "") == "store"
            ):
                return graph_node
        for graph_node in current_graph.nodes:
            if (
                graph_node.op == "call_function"
                and getattr(graph_node.target, "__name__", "") == "_for_loop"
            ):
                stack.append((device_ir.graphs[graph_node.args[0]].graph, depth + 1))
    return None


def _resolve_multi_block_ids(
    ctx: BuildContext,
    body_graph: torch.fx.Graph,
    block_ids: list[int],
    upper_bounds: list,
) -> list[int]:
    """Disambiguate reused block ids on a combined multi-dim ``_for_loop`` node.

    A single ``for tm, tp in hl.tile([bm, np])`` statement produces one
    ``_for_loop`` node whose ``block_ids`` can all be the same reused
    placeholder id, with the real per-dimension identities living in the
    body's own tile symbols. Since several new dimensions are introduced at
    once here, disambiguate by matching each dimension's declared upper
    bound against each candidate block's real size hint.
    """
    from ..support import block_id_from_key

    candidates = {
        info[0]
        for body_node in body_graph.nodes
        if (info := ctx.node_symbol_info(body_node)) is not None
        and info[0] not in ctx.block_id_to_iv
    }
    candidates.update(
        cand_id
        for body_node in body_graph.nodes
        if getattr(body_node.target, "__name__", "") == "_get_symnode"
        and body_node.args
        and (cand_id := block_id_from_key(body_node.args[0])) is not None
        and cand_id not in ctx.block_id_to_iv
    )
    remaining = set(candidates)
    resolved: list[int] = []
    for bid, ub in zip(block_ids, upper_bounds, strict=True):
        if bid in remaining:
            resolved.append(bid)
            remaining.discard(bid)
            continue
        ub_static = ub if isinstance(ub, int) else None
        match: int | None = None
        if ub_static is not None:
            for cand in remaining:
                block_info = next(
                    (b for b in ctx.env.block_sizes if b.block_id == cand), None
                )
                if block_info is None:
                    continue
                try:
                    if int(block_info.size_hint()) == ub_static:
                        match = cand
                        break
                except (TypeError, ValueError):
                    continue
        if match is None and remaining:
            match = next(iter(remaining))
        resolved.append(match if match is not None else bid)
        if match is not None:
            remaining.discard(match)
    return resolved


def lower_nested_for_loop(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a (possibly multi-dimensional) nested scf.for loop with optional
    synthetic store, recursing one ``scf.for`` per block id to arbitrary depth.
    """
    from ..support import NodeLoweringError
    from ..support import block_id_from_key

    body_graph_id = node.args[0]
    block_ids = list(node.args[1])
    upper_bounds = list(node.args[2])
    iter_arg_nodes = list(node.args[3])
    assert len(block_ids) == len(upper_bounds)
    body_graph_info = ctx.host_function.device_ir.graphs[body_graph_id]
    body_graph = body_graph_info.graph

    if len(block_ids) == 1:
        block_id = block_ids[0]
        # Helion can reuse the enclosing grid block id on a nested loop node.
        # The body still contains the inner scalar symbol, whose origin is
        # authoritative.
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
            elif not body_block_ids:
                # Direct body is a pure wrapper with no symbols at this level
                # (loop nested 3+ levels deep); unwrap further nested
                # ``_for_loop`` wrappers to find the block id introduced here.
                resolved = _find_reused_block_id(ctx, body_graph)
                if resolved is not None:
                    block_id = resolved
        block_ids = [block_id]
    else:
        if iter_arg_nodes:
            # Defensive only: Helion's device IR never attaches a carried
            # accumulator directly to a combined multi-dim tile's own
            # ``_for_loop`` node (verified empirically) — every dimension in
            # a combined ``hl.tile([a, b])`` is parallel by construction, and
            # a genuine reduction always gets its own separate, single-block
            # ``_for_loop`` nested inside (fully supported, see
            # ``_find_descendant_store``). Host tensors read inside a
            # combined tile are re-materialized via ``_host_tensor`` and never
            # lifted as iter args either. If this ever fires, Helion's IR
            # shape changed and the recursive emitter below needs to thread
            # ``iter_arg_nodes`` through every level, not just the innermost.
            raise NodeLoweringError(
                node,
                reason=(
                    "Combined multi-dimensional tile loops with an external "
                    "loop-carried accumulator are not supported"
                ),
                recovery_hint=(
                    "Split the combined hl.tile([...]) into separate nested "
                    "hl.tile() loops, or move the accumulator to an inner loop"
                ),
            )
        block_ids = _resolve_multi_block_ids(ctx, body_graph, block_ids, upper_bounds)

    return _emit_for_loop_level(
        ctx, node, body_graph, block_ids, upper_bounds, iter_arg_nodes, 0
    )


def _compute_synthetic_tile_geometry(
    ctx: BuildContext,
    *,
    full_shape: list[int],
    index_nodes: list | tuple,
    dim_block_ids: list[int | None],
    inner_dim: int,
    block_id: int,
    active_outer_block_ids: set[int],
    is_grid_loop: bool,
    ub_static: int | None,
    step: int,
) -> tuple[list[int], list[ir.Value]]:
    """Compute a synthetic per-iteration accumulator's shape and the offsets
    it flushes at, one entry per destination-store dimension.

    For the loop's own dimension (``inner_dim``), the tile spans the whole
    loop range. For a scalar-indexed dimension, the tile has size 1 at that
    scalar's current value. For a dimension owned by an active outer loop,
    the tile spans that outer loop's block size at its current offset. Any
    remaining dimension without a resolvable block id falls back to the
    nearest other active outer loop's block id (grid loops only), or is left
    unreduced at its full declared size with a zero offset.
    """
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
                ctx.block_id_to_iv.get(dim_bid) if isinstance(dim_bid, int) else None
            )
            if scalar_value is None:
                scalar_value = ctx.get_value(idx_node)
            flush_offsets.append(
                scalar_value if scalar_value is not None else ctx.index_const(0)
            )
            continue
        if (
            isinstance(dim_bid, int)
            and dim_bid in active_outer_block_ids
            and dim_bid in ctx.block_id_to_size
        ):
            tile_shape.append(ctx.block_id_to_size[dim_bid])
            flush_offsets.append(ctx.block_id_to_iv[dim_bid])
        elif (
            dim_bid is None
            and idx_node is not None
            and not isinstance(idx_node, slice)
            and not is_grid_loop
            and fallback_outer_bid is not None
            and fallback_outer_bid in ctx.block_id_to_size
        ):
            tile_shape.append(ctx.block_id_to_size[fallback_outer_bid])
            flush_offsets.append(ctx.block_id_to_iv[fallback_outer_bid])
        else:
            tile_shape.append(int(dim_size))
            flush_offsets.append(ctx.index_const(0))
    return tile_shape, flush_offsets


def _resolve_loop_upper_bound(
    ctx: BuildContext,
    node: torch.fx.Node,
    ub_src: object,
) -> tuple[int | None, ir.Value | None]:
    """Resolve a ``_for_loop`` upper bound to a static int and/or an ir.Value."""
    import mlir.ir as ir
    import torch
    import torch.fx

    from ..support import NodeLoweringError

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
                except (TypeError, ValueError):
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
    return ub_static, ub_val


def _emit_for_loop_level(
    ctx: BuildContext,
    node: torch.fx.Node,
    body_graph: torch.fx.Graph,
    block_ids: list[int],
    upper_bounds: list,
    iter_arg_nodes: list,
    level: int,
) -> ir.Value:
    """Emit one ``scf.for`` for ``block_ids[level]``.

    Only the innermost level (``level == len(block_ids) - 1``) actually
    lowers ``body_graph``'s content; outer levels recurse into the next
    level and thread that level's synthetic accumulator (if any) through via
    ``ctx.push_store_ctx``, exactly like naturally-nested ``_for_loop`` FX
    nodes already do (see ``_find_descendant_store``). This lets a single
    multi-dimensional ``_for_loop`` node (e.g. combined ``hl.tile([m, n])``)
    lower to nested ``scf.for`` loops, one per dimension.
    """
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir
    import torch
    import torch.fx

    from ..support import NodeLoweringError
    from ..support import torch_dtype_to_mlir

    block_id = block_ids[level]
    ub_src = upper_bounds[level]
    is_innermost = level == len(block_ids) - 1
    ub_static, ub_val = _resolve_loop_upper_bound(ctx, node, ub_src)

    body_scalar_kinds = {
        info[1]
        for body_node in body_graph.nodes
        if (info := ctx.node_symbol_info(body_node)) is not None
    }
    # A pure-wrapper body (loop nested 3+ levels deep) has no direct symbol
    # references to inspect, so also fall back to the block's own step size:
    # grid loops always have unit step by construction.
    is_grid_loop = (
        "grid" in body_scalar_kinds or ctx.block_id_to_size.get(block_id) == 1
    )
    step = ctx.block_id_to_size.get(block_id, ub_static if ub_static is not None else 1)

    if is_innermost:
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
    else:
        out_args = []
        invariant_pairs = []
        carried_pairs = []
        iter_init_vals = []

    active_outer_block_ids = set(ctx.block_id_to_iv.keys())
    synthetic_store_ctx: ForStoreContext | None = None
    synthetic_iter_index: int | None = None
    # ``body_graph`` is shared by every level of a combined multi-dim
    # ``_for_loop`` node, and an intermediate level of a naturally-nested
    # chain (e.g. the middle loop of ``grid -> grid -> tile``) has no store
    # of its own; either way, searching descendants finds the store that
    # this level's accumulator (if any) must eventually flush into.
    store_node = _find_descendant_store(ctx, body_graph)
    if store_node is not None:
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
            dim_block_ids: list[int | None] = []
            inner_dim: int | None = None
            for dim, idx_node in enumerate(index_nodes):
                if dim >= rank:
                    break
                dim_bid = ctx.infer_block_id_from_index(idx_node)
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
                tile_shape, flush_offsets = _compute_synthetic_tile_geometry(
                    ctx,
                    full_shape=full_shape,
                    index_nodes=index_nodes,
                    dim_block_ids=dim_block_ids,
                    inner_dim=inner_dim,
                    block_id=block_id,
                    active_outer_block_ids=active_outer_block_ids,
                    is_grid_loop=is_grid_loop,
                    ub_static=ub_static,
                    step=step,
                )
                tile_empty = tensor_d.EmptyOp(tile_shape, elem_ty).result
                if isinstance(elem_ty, ir.FloatType):
                    zero_attr = ir.FloatAttr.get(elem_ty, 0.0)
                else:
                    zero_attr = ir.IntegerAttr.get(elem_ty, 0)
                zero = arith_d.ConstantOp(elem_ty, zero_attr).result
                tile_init = linalg_d.fill(zero, outs=[tile_empty])
                synthetic_iter_index = len(iter_init_vals)
                iter_init_vals.append(tile_init)

                from .for_store_context import ForStoreContext

                synthetic_store_ctx = ForStoreContext(flush_offsets=flush_offsets)
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
        if is_innermost:
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
                carried_placeholders,
                body_block.arguments[1 : 1 + len(carried_pairs)],
                strict=False,
            ):
                ctx.set_value(ph_node, body_arg)
            if synthetic_store_ctx is not None and synthetic_iter_index is not None:
                synthetic_store_ctx.current = body_block.arguments[
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
        else:
            if synthetic_store_ctx is not None and synthetic_iter_index is not None:
                synthetic_store_ctx.current = body_block.arguments[
                    1 + synthetic_iter_index
                ]
                with ctx.push_store_ctx(synthetic_store_ctx):
                    _emit_for_loop_level(
                        ctx,
                        node,
                        body_graph,
                        block_ids,
                        upper_bounds,
                        iter_arg_nodes,
                        level + 1,
                    )
            else:
                _emit_for_loop_level(
                    ctx,
                    node,
                    body_graph,
                    block_ids,
                    upper_bounds,
                    iter_arg_nodes,
                    level + 1,
                )
            yield_vals = []
        if synthetic_store_ctx is not None:
            current = synthetic_store_ctx.current
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
            parent_current = parent_ctx.current
            if parent_current is not None:
                parent_type = ir.RankedTensorType(parent_current.type)
                tile_type = ir.RankedTensorType(final_tile.type)
                offsets = list(synthetic_store_ctx.flush_offsets)
                if len(offsets) != parent_type.rank:
                    offsets = offsets[: parent_type.rank]
                    offsets.extend(
                        ctx.index_const(0)
                        for _ in range(parent_type.rank - len(offsets))
                    )
                # An ancestor's own induction variable is only a valid offset
                # here if the parent accumulator's dimension actually spans
                # its full range; if that dimension has already been reduced
                # to a single local slot (size 1, e.g. a grid ancestor two or
                # more levels up), the offset must be 0 or the insert goes
                # out of bounds.
                offsets = [
                    ctx.index_const(0) if int(parent_type.shape[d]) == 1 else off
                    for d, off in enumerate(offsets)
                ]
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
                parent_ctx.current = updated
        else:
            ctx.forall_insert_slices.append(
                (final_tile, synthetic_store_ctx.flush_offsets, None)
            )
    return for_op
