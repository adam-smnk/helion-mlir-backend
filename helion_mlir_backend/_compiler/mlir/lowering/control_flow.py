"""Outer parallel control-flow lowering for MLIR kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
from mlir.dialects import affine as affine_d
from mlir.dialects import scf as scf_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from ..analysis.geometry import is_loop_node
from . import emit
from .registry import lowers

if TYPE_CHECKING:
    import torch

    from ..analysis.geometry import LoopBounds
    from ..build_context import BuildContext
    from .for_store_context import ForStoreContext


@lowers(tracing_ops._new_var)
def lower_new_var(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    return ctx.get_value(node.args[0])


@lowers(tracing_ops._phi)
def lower_phi(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``_phi(before, after)``: the loop result (``after``) replaces the value."""
    return ctx.get_value(node.args[1])


def _block_id_to_out_dim_from_terminal_store(
    ctx: BuildContext,
    grid_block_ids: list[int],
    graphs: list[torch.fx.Graph] | None = None,
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
    store is found. Scoped to *graphs* when given (e.g. one phase's own
    graphs), otherwise scans the whole kernel's.
    """
    from ..support.index_meta import resolve_index_descriptor

    expected_block_ids = set(grid_block_ids)
    search_graphs = (
        graphs
        if graphs is not None
        else [gi.graph for gi in ctx.host_function.device_ir.graphs]
    )

    for graph in search_graphs:
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is not memory_ops.store:
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


def _tile_offset(
    ctx: BuildContext, trip_iv: ir.Value, begin: int, step: int
) -> ir.Value:
    """Absolute tile offset ``begin + trip_iv * step`` for a normalized forall IV."""

    if begin == 0 and step == 1:
        return trip_iv
    expr = ir.AffineExpr.get_add(
        ir.AffineExpr.get_mul(ir.AffineDimExpr.get(0), ir.AffineConstantExpr.get(step)),
        ir.AffineConstantExpr.get(begin),
    )
    affine_map = ir.AffineMap.get(1, 0, [expr])
    return affine_d.AffineApplyOp(affine_map, [trip_iv]).result


def build_kernel_body(
    ctx: BuildContext,
    out_tensors: list[torch.Tensor],
    root_ids: list[int] | None = None,
    grid_block_id_groups: list[list[int]] | None = None,
) -> list[ir.Value]:
    """Build the outer ``scf.forall`` and its parallel insert terminator.

    ``out_tensors`` is every tensor the active phase writes (usually one; more
    than one when a single phase stores into several distinct output
    tensors). All of them share this forall's single iteration space, so
    their tiled dimensions must agree -- see the shape-compatibility check
    below.

    ``root_ids``/``grid_block_id_groups`` scope this build to one
    ``hl.barrier()``-separated phase's own root graphs/block-id groups
    instead of the whole kernel's (the default, when both are ``None``).

    Maps each grid block_id to its actual destination dimension (not just positional).
    """

    from ..support import torch_dtype_to_mlir

    primary_tensor = out_tensors[0]
    out_shape = [int(dim) for dim in primary_tensor.shape]

    # ``grid_block_ids`` groups block ids by the outer ``for`` statement that
    # produced them (e.g. a single ``for tile_m, tile_n in hl.tile([m, n])``
    # yields one entry ``[0, 1]``), so the flattened list must advance per
    # block id, not per group, or every block id in a multi-dim statement
    # collapses onto one dimension.
    groups = (
        grid_block_id_groups
        if grid_block_id_groups is not None
        else ctx.host_function.device_ir.grid_block_ids
    )
    grid_block_ids_flat: list[int] = []
    for ids in groups:
        grid_block_ids_flat.extend(ids)

    phase_graphs = _phase_graph_closure(ctx, root_ids) if root_ids is not None else None

    # Prefer the authoritative mapping derived from the terminal store's own
    # index expression: loop declaration order does not necessarily match the
    # order block ids are indexed in the output (e.g. ``out[tm, panel, :]``
    # with ``panel``'s loop declared before ``tm``'s). Fall back to loop
    # declaration order only if no matching terminal store is found.
    block_id_to_out_dim = _block_id_to_out_dim_from_terminal_store(
        ctx, grid_block_ids_flat, graphs=phase_graphs
    )
    if block_id_to_out_dim is None:
        block_id_to_out_dim = {
            block_id: out_dim for out_dim, block_id in enumerate(grid_block_ids_flat)
        }

    for idx, bid in enumerate(grid_block_ids_flat):
        if block_id_to_out_dim.get(bid, idx) >= len(out_shape):
            from ..support import UnsupportedOperationError

            raise UnsupportedOperationError(
                "independent top-level loops with incompatible geometry",
                reason=(
                    f"block_id {bid} needs output dimension "
                    f"{block_id_to_out_dim.get(bid, idx)}, but the resolved "
                    f"output tensor only has {len(out_shape)} dimension(s); "
                    "this usually means two or more independent top-level "
                    "hl.tile()/hl.grid() loops (not separated by hl.barrier()) "
                    "iterate over incompatible shapes"
                ),
                alternatives=[
                    "separate independent loops with hl.barrier()",
                    (
                        "ensure all top-level loops in one phase share the same "
                        "iteration space"
                    ),
                ],
            )
    geometry = ctx.geometry
    root_bounds = [geometry.root_bounds[bid] for bid in grid_block_ids_flat]
    trip_counts = [
        -(-(bounds.end - bounds.begin) // bounds.step) for bounds in root_bounds
    ]

    if len(out_tensors) > 1:
        _validate_multi_output_shapes(out_tensors, out_shape, block_id_to_out_dim)

    # The outer scf.forall emits one statically-sized extract/insert per
    # iteration (no per-iteration dynamic clamp for a ragged last tile), so a
    # dimension that's part of a COMBINED multi-dim tile (e.g. hl.tile([m, n]))
    # and needs more than one iteration must divide evenly by its block size;
    # a ragged last iteration would read/write past the tensor's real bound.
    # A single-dimension hl.tile() is unaffected: its own tile.end/mask-based
    # dynamic clamping (see tile_index_ops.scalar_tile_value) already handles
    # raggedness correctly. A single iteration (step >= extent) is also
    # unaffected since slice_plan already clamps that case statically.
    combined_block_ids = {bid for ids in groups if len(ids) > 1 for bid in ids}
    for block_id, bounds, trips in zip(
        grid_block_ids_flat, root_bounds, trip_counts, strict=True
    ):
        if (
            block_id in combined_block_ids
            and trips > 1
            and geometry.is_ragged(block_id)
        ):
            from ..support import UnsupportedOperationError

            raise UnsupportedOperationError(
                "ragged combined-tile block size",
                reason=(
                    f"block_id {block_id}: dimension of size "
                    f"{bounds.end - bounds.begin} is not evenly divisible by "
                    f"block size {bounds.step}, and needs more than one "
                    "iteration; this backend does not yet support a "
                    "dynamically-sized boundary tile in this position"
                ),
                alternatives=[
                    "choose a block size that evenly divides this dimension",
                    "restructure the kernel so this dimension needs only one iteration",
                ],
            )

    output_search_graphs = (
        phase_graphs
        if phase_graphs is not None
        else [gi.graph for gi in ctx.host_function.device_ir.graphs]
    )
    output_emptys = [
        _existing_output_value(ctx, tensor, output_search_graphs)
        or tensor_d.EmptyOp(
            [int(d) for d in tensor.shape],
            torch_dtype_to_mlir(tensor.dtype),
        ).result
        for tensor in out_tensors
    ]
    rank = len(grid_block_ids_flat)
    forall = scf_d.ForallOp(
        [0] * rank, trip_counts, [1] * rank, shared_outs=output_emptys
    )

    with ir.InsertionPoint(forall.body):
        # Bound for the whole compile (no enclosing scope to restore to),
        # unlike nested scf.for levels which use ctx.enter_for_loop's
        # save/restore.
        for block_id, bounds, trip_iv in zip(
            grid_block_ids_flat, root_bounds, forall.induction_variables, strict=True
        ):
            ctx.block_id_to_trip_iv[block_id] = trip_iv
            ctx.block_id_to_iv[block_id] = _tile_offset(
                ctx, trip_iv, bounds.begin, bounds.step
            )
            ctx.block_id_to_bounds[block_id] = (bounds.begin, bounds.end)
        shared_outs = list(forall.inner_iter_args)
        ctx.lower_root_graphs(shared_outs[0], root_ids=root_ids)
        in_parallel = scf_d.InParallelOp()
        with ir.InsertionPoint(in_parallel.block):
            tensor_id_to_index = {id(tensor): i for i, tensor in enumerate(out_tensors)}
            for value, offsets, *rest in ctx.forall_insert_slices:
                static_sizes_field = rest[0] if rest else None
                target_tensor_id = rest[1] if len(rest) > 1 else None
                source_type = ir.RankedTensorType(value.type)
                static_sizes = (
                    list(static_sizes_field)
                    if static_sizes_field
                    else list(source_type.shape)
                )
                rank = len(static_sizes)
                destination = shared_outs[0]
                if len(shared_outs) > 1:
                    index = tensor_id_to_index.get(target_tensor_id)
                    if index is None:
                        from ..support import UnsupportedOperationError

                        raise UnsupportedOperationError(
                            "multi-output store routing",
                            reason=(
                                "could not determine which of the kernel's output "
                                "tensors this store targets (e.g. a nested-reduction "
                                "accumulator flushed into a multi-output phase is "
                                "not yet supported)"
                            ),
                            alternatives=[
                                (
                                    "avoid nested hl.tile() reductions in a phase "
                                    "that returns multiple tensors"
                                )
                            ],
                        )
                    destination = shared_outs[index]
                _validate_parallel_insert_fits(
                    static_sizes, ir.RankedTensorType(destination.type)
                )
                tensor_d.ParallelInsertSliceOp(
                    value,
                    destination,
                    offsets,
                    [],
                    [],
                    static_offsets=[ir.ShapedType.get_dynamic_size()] * rank,
                    static_sizes=static_sizes,
                    static_strides=[1] * rank,
                )

    return list(forall.results)


def _validate_parallel_insert_fits(
    static_sizes: list[int], destination_type: ir.RankedTensorType
) -> None:
    """Reject a terminal insert that would write past the output tensor.

    Catches wrong tile geometry (e.g. a synthetic accumulator sized from the
    wrong loop's extent) before it becomes an out-of-bounds write at runtime.
    """
    from ..support import UnsupportedOperationError

    dest_shape = [int(d) for d in destination_type.shape]
    if len(dest_shape) != len(static_sizes):
        return
    for dim, (size, extent) in enumerate(zip(static_sizes, dest_shape, strict=True)):
        if int(size) <= extent:
            continue
        raise UnsupportedOperationError(
            "output tile larger than the destination dimension",
            reason=(
                f"writing a tile of size {int(size)} into dimension {dim} of an "
                f"output whose extent is only {extent} (tile sizes "
                f"{[int(s) for s in static_sizes]}, output shape {dest_shape}); "
                "the per-iteration tile geometry does not match the output"
            ),
            alternatives=[
                "index the output in the same order the value is computed",
                "reorder the value explicitly with .permute() before storing",
            ],
        )


def _existing_output_value(
    ctx: BuildContext,
    tensor: torch.Tensor,
    graphs: list[torch.fx.Graph],
) -> ir.Value | None:
    """The already-lowered value for *tensor*, if some earlier node already
    produced it (e.g. ``torch.zeros``/``torch.full``) in the same phase, or
    if it is itself one of this phase's own function parameters (a value an
    earlier phase already computed and the multi-phase driver threaded in by
    name -- see ``bound_kernel.py``/``phase_plan.py``).

    A tile loop's ``shared_outs`` init must preserve whatever the host-level
    output variable already held, not unconditionally reset it via a fresh
    ``tensor.empty()`` -- that silently discards real initialization (e.g.
    zero-padding written before a loop that only covers part of the tensor,
    or by an earlier phase entirely), replacing it with uninitialized memory
    for anything the loop doesn't touch.
    """
    from ..phase_plan import resolve_host_variable_name

    name = resolve_host_variable_name(ctx.host_function, tensor)
    if name is not None and name in ctx.param_to_value:
        return ctx.param_to_value[name]

    # Matched by object identity against each node's traced fake value,
    # which is how the same underlying tensor is tracked across FX nodes.
    for graph in graphs:
        for node in graph.nodes:
            if node.meta.get("val") is tensor:
                value = ctx.node_to_value.get(node)
                if value is not None:
                    return value
    return None


def _phase_graph_closure(
    ctx: BuildContext, root_ids: list[int]
) -> list[torch.fx.Graph]:
    """Every graph belonging to phase-root graph IDs, including nested loops.

    Terminal stores usually reside in a nested `_for_loop` body rather than
    directly in a root graph, so a root-only scan would miss the authoritative
    index order and incorrectly use declaration order as a fallback.
    """
    device_ir = ctx.host_function.device_ir
    graphs: list[torch.fx.Graph] = []
    seen_ids: set[int] = set()
    pending = list(root_ids)
    while pending:
        graph_id = pending.pop()
        if graph_id in seen_ids:
            continue
        seen_ids.add(graph_id)
        graph = device_ir.graphs[graph_id].graph
        graphs.append(graph)
        pending.extend(
            node.args[0]
            for node in graph.nodes
            if node.op == "call_function" and is_loop_node(node)
        )
    return graphs


def _validate_multi_output_shapes(
    out_tensors: list[torch.Tensor],
    primary_shape: list[int],
    block_id_to_out_dim: dict[int, int],
) -> None:
    """Reject a phase whose N output tensors disagree on a tiled dimension.

    All outputs of one phase share a single ``scf.forall`` iteration space,
    so every tiled dimension's extent must match across them; a mismatch
    means the outputs need separate loops (separate phases), which V1 does
    not yet derive automatically.
    """
    from ..support import UnsupportedOperationError

    for extra in out_tensors[1:]:
        extra_shape = [int(dim) for dim in extra.shape]
        for dim in block_id_to_out_dim.values():
            if dim < len(extra_shape) and dim < len(primary_shape):
                if extra_shape[dim] != primary_shape[dim]:
                    raise UnsupportedOperationError(
                        "differently-shaped multi-output kernel",
                        reason=(
                            "all tensors written within a single hl.tile()/"
                            "hl.grid() loop must share the same tiled shape; got "
                            f"{primary_shape} and {extra_shape}"
                        ),
                        alternatives=[
                            (
                                "split differently-shaped outputs into separate "
                                "hl.tile()/hl.grid() loops (separated by "
                                "hl.barrier() if one depends on the other)"
                            )
                        ],
                    )


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
                and graph_node.target is memory_ops.store
            ):
                return graph_node
        for graph_node in current_graph.nodes:
            if graph_node.op == "call_function" and is_loop_node(graph_node):
                stack.append((device_ir.graphs[graph_node.args[0]].graph, depth + 1))
    return None


@lowers(tracing_ops._for_loop, tracing_ops._for_loop_step)
def lower_nested_for_loop(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a (possibly multi-dimensional) nested ``_for_loop``/``_for_loop_step``
    to one ``scf.for`` per block id, with an optional synthetic store.

    Block ids come from the loop's ``ForLoopGraphInfo``; bounds and steps from
    the node itself (``_for_loop(graph_id, begin, end, args[, step])``).
    """
    from ..analysis.geometry import loop_block_ids
    from ..support import NodeLoweringError

    iter_arg_nodes = list(node.args[3])
    body_graph = ctx.host_function.device_ir.graphs[node.args[0]].graph
    block_ids = loop_block_ids(ctx.host_function, node)
    loop_bounds = ctx.geometry.loop_bounds(node, block_ids)

    if len(block_ids) > 1 and iter_arg_nodes:
        # Helion never attaches a carried accumulator to a combined multi-dim
        # tile's own ``_for_loop`` node: every dimension of ``hl.tile([a, b])``
        # is parallel, and reductions get their own nested single-block loop.
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

    return _emit_for_loop_level(
        ctx, node, body_graph, block_ids, loop_bounds, iter_arg_nodes, 0
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
    begin_static: int | None,
    end_static: int | None,
    step: int,
) -> tuple[list[int], list[ir.Value]]:
    """Compute a synthetic per-iteration accumulator's shape and the offsets
    it flushes at, one entry per destination-store dimension.

    For the loop's own dimension (``inner_dim``), the tile spans ``[0, end)``
    and flushes at ``begin`` (see ``ForStoreContext.flush_window``). For a
    scalar-indexed dimension, the tile has size 1 at that
    scalar's current value. For a dimension owned by an active outer loop,
    the tile spans that outer loop's block size at its current offset. Any
    remaining dimension without a resolvable block id falls back to the
    nearest other active outer loop's block id (grid loops only), or is left
    unreduced at its full declared size with a zero offset.
    """
    geometry = ctx.geometry
    tile_shape: list[int] = []
    flush_offsets: list[ir.Value] = []
    outer_bids = [bid for bid in active_outer_block_ids if bid != block_id]
    fallback_outer_bid = outer_bids[0] if outer_bids else None
    for dim, dim_size in enumerate(full_shape):
        idx_node = index_nodes[dim] if dim < len(index_nodes) else None
        dim_bid = dim_block_ids[dim]
        if dim == inner_dim or dim_bid == block_id:
            tile_shape.append(end_static if end_static is not None else step)
            flush_offsets.append(ctx.index_const(begin_static or 0))
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
            and dim_bid in geometry.blocks
        ):
            tile_shape.append(geometry.tile_extent(dim_bid))
            flush_offsets.append(ctx.block_id_to_iv[dim_bid])
        elif (
            dim_bid is None
            and idx_node is not None
            and not isinstance(idx_node, slice)
            and not is_grid_loop
            and fallback_outer_bid is not None
            and fallback_outer_bid in geometry.blocks
        ):
            tile_shape.append(geometry.tile_extent(fallback_outer_bid))
            flush_offsets.append(ctx.block_id_to_iv[fallback_outer_bid])
        else:
            tile_shape.append(int(dim_size))
            flush_offsets.append(ctx.index_const(0))
    return tile_shape, flush_offsets


def _resolve_loop_bound(
    ctx: BuildContext,
    node: torch.fx.Node,
    source: object,
) -> int | ir.Value:
    """Resolve a ``_for_loop`` begin/end to a static int or an index value."""
    import torch
    import torch.fx

    from ..support import NodeLoweringError

    if isinstance(source, int):
        return int(source)
    if isinstance(source, torch.SymInt):
        try:
            return int(source)
        except (TypeError, ValueError):
            pass
    value: ir.Value | None = None
    if isinstance(source, torch.fx.Node):
        value = ctx.get_value(source)
        if value is None:
            meta_val = source.meta.get("val")
            if isinstance(meta_val, int):
                return int(meta_val)
    elif isinstance(source, ir.Value):
        value = source
    if value is None:
        raise NodeLoweringError(
            node,
            reason=f"Unsupported loop bound: {source!r}",
            recovery_hint="Ensure loop bounds are integer constants or scalar tensor values",
        )
    return ctx.cast_to_index(value)


def _prepare_synthetic_accumulator(
    ctx: BuildContext,
    body_graph: torch.fx.Graph,
    block_id: int,
    active_outer_block_ids: set[int],
    is_grid_loop: bool,
    begin_static: int | None,
    end_static: int | None,
    step: int,
    iter_init_vals: list[ir.Value],
) -> tuple[ForStoreContext | None, int | None]:
    """Discover a descendant store and, if this level owns part of its
    destination, allocate + zero-fill a synthetic accumulator tile for it.

    ``body_graph`` is shared by every level of a combined multi-dim
    ``_for_loop`` node, and an intermediate level of a naturally-nested chain
    (e.g. the middle loop of ``grid -> grid -> tile``) has no store of its
    own; either way, searching descendants finds the store that this level's
    accumulator (if any) must eventually flush into. On success, appends the
    zero-filled tile to ``iter_init_vals`` (mutated in place) and returns its
    ``ForStoreContext`` plus its index within ``iter_init_vals``; returns
    ``(None, None)`` if no such store/geometry was found.
    """
    import torch

    from ..support import torch_dtype_to_mlir

    store_node = _find_descendant_store(ctx, body_graph)
    if store_node is None:
        return None, None

    target_node = store_node.args[0]
    index_nodes = store_node.args[1]
    value_node = store_node.args[2]
    target_val = ctx.get_value(target_node)
    target_meta = (
        target_node.meta.get("val") if isinstance(target_node, torch.fx.Node) else None
    )
    value_meta = (
        value_node.meta.get("val") if isinstance(value_node, torch.fx.Node) else None
    )
    target_type: ir.RankedTensorType | None = None
    target_rank_matches = True
    if target_val is not None:
        target_type = ir.RankedTensorType(target_val.type)
        target_rank_matches = target_type.rank == len(index_nodes)
    if not (isinstance(index_nodes, (list, tuple)) and target_rank_matches):
        return None, None

    if target_val is not None:
        assert target_type is not None
        full_shape = [1] * len(index_nodes)
        for dim, idx_node in enumerate(index_nodes):
            if dim < len(target_type.shape) and not ctx.is_scalar_index_node(idx_node):
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
    if inner_dim is None:
        return None, None

    owns_inner_dim = dim_block_ids[inner_dim] == block_id
    if owns_inner_dim and begin_static is None:
        from ..support import UnsupportedOperationError

        raise UnsupportedOperationError(
            "store inside a loop with a runtime begin",
            reason=(
                f"block_id {block_id}'s loop writes its own output dimension but "
                "starts at a runtime offset"
            ),
        )

    tile_shape, flush_offsets = _compute_synthetic_tile_geometry(
        ctx,
        full_shape=full_shape,
        index_nodes=index_nodes,
        dim_block_ids=dim_block_ids,
        inner_dim=inner_dim,
        block_id=block_id,
        active_outer_block_ids=active_outer_block_ids,
        is_grid_loop=is_grid_loop,
        begin_static=begin_static if owns_inner_dim else 0,
        end_static=end_static,
        step=step,
    )
    flush_window = None
    if owns_inner_dim and begin_static and end_static is not None:
        flush_window = (inner_dim, begin_static, end_static - begin_static)
    tile_init = emit.filled(tile_shape, elem_ty, 0)
    synthetic_iter_index = len(iter_init_vals)
    iter_init_vals.append(tile_init)

    from .for_store_context import ForStoreContext

    target_tensor_id = (
        id(target_meta) if isinstance(target_meta, torch.Tensor) else None
    )
    return (
        ForStoreContext(
            flush_offsets=flush_offsets,
            target_tensor_id=target_tensor_id,
            flush_window=flush_window,
        ),
        synthetic_iter_index,
    )


def _lower_innermost_loop_body(
    ctx: BuildContext,
    body_graph: torch.fx.Graph,
    body_block: ir.Block,
    out_args: list,
    invariant_pairs: list[tuple],
    carried_pairs: list[tuple],
    iter_pairs: list[tuple],
    synthetic_store_ctx: ForStoreContext | None,
    synthetic_iter_index: int | None,
) -> list[ir.Value]:
    """Bind placeholders, lower ``body_graph``'s real content, and collect
    the values to yield back to the enclosing ``scf.for``."""
    import torch

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
        synthetic_store_ctx.current = body_block.arguments[1 + synthetic_iter_index]
        with ctx.push_store_ctx(synthetic_store_ctx):
            ctx.lower_graph(body_graph)
    else:
        ctx.lower_graph(body_graph)
    yield_vals = []
    for a in out_args:
        v = ctx.get_value(a) if isinstance(a, torch.fx.Node) else None
        if v is not None:
            yield_vals.append(v)
    return yield_vals


def _lower_outer_loop_level(
    ctx: BuildContext,
    node: torch.fx.Node,
    body_graph: torch.fx.Graph,
    block_ids: list[int],
    loop_bounds: list[LoopBounds],
    iter_arg_nodes: list,
    level: int,
    body_block: ir.Block,
    synthetic_store_ctx: ForStoreContext | None,
    synthetic_iter_index: int | None,
) -> list[ir.Value]:
    """Recurse into the next nested ``scf.for`` level for a non-innermost
    dimension of a combined multi-dim ``_for_loop`` node. Never yields any
    values of its own; only a synthetic accumulator (added by the caller)
    may be threaded through this level."""
    if synthetic_store_ctx is not None and synthetic_iter_index is not None:
        synthetic_store_ctx.current = body_block.arguments[1 + synthetic_iter_index]
        with ctx.push_store_ctx(synthetic_store_ctx):
            _emit_for_loop_level(
                ctx,
                node,
                body_graph,
                block_ids,
                loop_bounds,
                iter_arg_nodes,
                level + 1,
            )
    else:
        _emit_for_loop_level(
            ctx, node, body_graph, block_ids, loop_bounds, iter_arg_nodes, level + 1
        )
    return []


def _emit_for_loop_level(
    ctx: BuildContext,
    node: torch.fx.Node,
    body_graph: torch.fx.Graph,
    block_ids: list[int],
    loop_bounds: list[LoopBounds],
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

    from ..support import NodeLoweringError

    block_id = block_ids[level]
    bounds = loop_bounds[level]
    is_innermost = level == len(block_ids) - 1
    begin = _resolve_loop_bound(ctx, node, bounds.begin)
    end = _resolve_loop_bound(ctx, node, bounds.end)
    is_grid_loop = ctx.geometry.is_grid(block_id)
    step = bounds.step

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
    synthetic_store_ctx, synthetic_iter_index = _prepare_synthetic_accumulator(
        ctx,
        body_graph,
        block_id,
        active_outer_block_ids,
        is_grid_loop,
        begin if isinstance(begin, int) else None,
        end if isinstance(end, int) else None,
        step,
        iter_init_vals,
    )
    for_op = scf_d.ForOp(
        ctx.as_index(begin),
        ctx.as_index(end),
        ctx.index_const(step),
        iter_args=iter_init_vals,
    )
    body_block = for_op.body
    with (
        ir.InsertionPoint(body_block),
        ctx.enter_for_loop(block_id, body_block.arguments[0], (begin, end)),
    ):
        if is_innermost:
            yield_vals = _lower_innermost_loop_body(
                ctx,
                body_graph,
                body_block,
                out_args,
                invariant_pairs,
                carried_pairs,
                iter_pairs,
                synthetic_store_ctx,
                synthetic_iter_index,
            )
        else:
            yield_vals = _lower_outer_loop_level(
                ctx,
                node,
                body_graph,
                block_ids,
                loop_bounds,
                iter_arg_nodes,
                level,
                body_block,
                synthetic_store_ctx,
                synthetic_iter_index,
            )
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
        _flush_synthetic_accumulator_to_parent(
            ctx, for_op, synthetic_store_ctx, synthetic_iter_index
        )
    return for_op


def _flush_synthetic_accumulator_to_parent(
    ctx: BuildContext,
    for_op: ir.Operation,
    synthetic_store_ctx: ForStoreContext,
    synthetic_iter_index: int,
) -> None:
    """Thread a completed loop level's synthetic accumulator result upward.

    If an enclosing level already has its own accumulator in progress
    (``ctx.for_store_ctx_stack``), insert this level's finished tile into it
    at ``flush_offsets``; otherwise this is the outermost accumulator, so
    record it for the top-level ``scf.forall``'s parallel-insert terminator.
    """

    final_tile = for_op.results[synthetic_iter_index]
    if synthetic_store_ctx.flush_window is not None:
        dim, begin, size = synthetic_store_ctx.flush_window
        shape = list(ir.RankedTensorType(final_tile.type).shape)
        offsets = [0] * len(shape)
        offsets[dim] = begin
        shape[dim] = size
        final_tile = tensor_d.ExtractSliceOp(
            ir.RankedTensorType.get(
                shape, ir.RankedTensorType(final_tile.type).element_type
            ),
            final_tile,
            [],
            [],
            [],
            static_offsets=offsets,
            static_sizes=shape,
            static_strides=[1] * len(shape),
        ).result
    if not ctx.for_store_ctx_stack:
        ctx.forall_insert_slices.append(
            (
                final_tile,
                synthetic_store_ctx.flush_offsets,
                None,
                synthetic_store_ctx.target_tensor_id,
            )
        )
        return

    parent_ctx = ctx.for_store_ctx_stack[-1]
    parent_current = parent_ctx.current
    if parent_current is None:
        return

    parent_type = ir.RankedTensorType(parent_current.type)
    tile_type = ir.RankedTensorType(final_tile.type)
    offsets = list(synthetic_store_ctx.flush_offsets)
    if len(offsets) != parent_type.rank:
        offsets = offsets[: parent_type.rank]
        offsets.extend(
            ctx.index_const(0) for _ in range(parent_type.rank - len(offsets))
        )
    # An ancestor's own induction variable is only a valid offset here if the
    # parent accumulator's dimension actually spans its full range; if that
    # dimension has already been reduced to a single local slot (size 1, e.g.
    # a grid ancestor two or more levels up), the offset must be 0 or the
    # insert goes out of bounds.
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
        static_offsets=[ir.ShapedType.get_dynamic_size()] * parent_type.rank,
        static_sizes=[int(dim) for dim in tile_type.shape],
        static_strides=[1] * parent_type.rank,
    ).result
    parent_ctx.current = updated
