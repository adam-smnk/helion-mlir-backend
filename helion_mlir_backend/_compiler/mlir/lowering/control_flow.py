"""Loop lowering: one ``scf.forall`` per root graph, ``scf.for`` for nested loops.

Written host tensors are threaded as SSA values (``tensor_state``): a forall
iteration works on its owned region of each written tensor and inserts it back
in the terminator; a nested ``scf.for`` carries the tensors its body writes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import affine as affine_d
from mlir.dialects import scf as scf_d
import mlir.ir as ir
import torch

from ..analysis.geometry import loop_block_ids
from ..analysis.tensor_effects import accessed_tensor
from ..support import NodeLoweringError
from . import emit
from .registry import lowers
from .slice_plan import tile_window
from .tensor_state import OwnedDim
from .tensor_state import owned_dims

if TYPE_CHECKING:
    from ..analysis.geometry import LoopBounds
    from ..build_context import BuildContext

log = logging.getLogger(__name__)


@lowers(tracing_ops._new_var)
def lower_new_var(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    return ctx.get_value(node.args[0])


@lowers(tracing_ops._phi)
def lower_phi(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``_phi(before, after)``: the loop result (``after``) replaces the value."""
    return ctx.get_value(node.args[1])


def build_phase_body(
    ctx: BuildContext, root_positions: list[int], outputs: tuple[str, ...]
) -> list[ir.Value]:
    """Lower the root graphs at ``root_positions`` in order; return ``outputs``' values.

    Every written host tensor (``outputs``) starts as its function argument.
    """
    device_ir = ctx.host_function.device_ir
    for name in outputs:
        ctx.tensors.bind(name, ctx.param_to_value[name])
    for position in root_positions:
        _lower_root(
            ctx, device_ir.root_ids[position], list(device_ir.grid_block_ids[position])
        )
    return [ctx.tensors.value(name) for name in outputs]


def _lower_root(ctx: BuildContext, graph_id: int, grid_ids: list[int]) -> None:
    """Parallel ``scf.forall`` if each iteration writes disjoint regions, else sequential."""
    geometry = ctx.geometry
    bounds = [geometry.root_bounds[block_id] for block_id in grid_ids]
    trips = [-(-(b.end - b.begin) // b.step) for b in bounds]

    names = list(ctx.effects.writes(graph_id))
    accesses = ctx.effects.accesses(graph_id)
    owned = {
        name: owned_dims(
            ctx,
            [node for node in accesses if accessed_tensor(node) == name],
            ir.RankedTensorType(ctx.tensors.value(name).type).rank,
            grid_ids,
        )
        for name in names
    }
    unowned = [
        (name, block_id)
        for name in names
        for block_id in grid_ids
        if block_id not in {dim.block_id for dim in owned[name].values()}
    ]
    graph = ctx.host_function.device_ir.graphs[graph_id].graph
    if not unowned:
        _emit_forall(ctx, graph, grid_ids, bounds, trips, names, owned)
        return
    log.debug(
        "root graph %d runs sequentially: %s is not partitioned by block_id %d",
        graph_id,
        *unowned[0],
    )
    _emit_sequential(ctx, graph, grid_ids, bounds, trips, names)


def _emit_forall(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    grid_ids: list[int],
    bounds: list[LoopBounds],
    trips: list[int],
    names: list[str],
    owned: dict[str, dict[int, OwnedDim]],
) -> None:
    rank = len(grid_ids)
    forall = scf_d.ForallOp(
        [0] * rank,
        trips,
        [1] * rank,
        shared_outs=[ctx.tensors.value(name) for name in names],
    )
    with ir.InsertionPoint(forall.body):
        for block_id, bound, trip_iv in zip(
            grid_ids, bounds, forall.induction_variables, strict=True
        ):
            _bind_grid_iv(ctx, block_id, bound, trip_iv)
        regions = {}
        for name, shared in zip(names, forall.inner_iter_args, strict=True):
            regions[name] = _owned_region(ctx, shared, owned[name])
            ctx.tensors.bind(
                name, emit.extract_slice(shared, *regions[name]), owned[name]
            )
        ctx.lower_graph(graph)
        in_parallel = scf_d.InParallelOp()
        with ir.InsertionPoint(in_parallel.block):
            for name, shared in zip(names, forall.inner_iter_args, strict=True):
                emit.parallel_insert_slice(
                    ctx.tensors.value(name), shared, *regions[name]
                )
    for name, result in zip(names, forall.results, strict=True):
        ctx.tensors.bind(name, result)


def _emit_sequential(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    grid_ids: list[int],
    bounds: list[LoopBounds],
    trips: list[int],
    names: list[str],
    level: int = 0,
) -> None:
    """One ``scf.for`` per grid dim carrying the full written tensors."""
    if level == len(grid_ids):
        ctx.lower_graph(graph)
        return
    for_op = scf_d.ForOp(
        ctx.index_const(0),
        ctx.index_const(trips[level]),
        ctx.index_const(1),
        iter_args=[ctx.tensors.value(name) for name in names],
    )
    with ir.InsertionPoint(for_op.body):
        _bind_grid_iv(ctx, grid_ids[level], bounds[level], for_op.induction_variable)
        for name, arg in zip(names, for_op.inner_iter_args, strict=True):
            ctx.tensors.rebind(name, arg)
        _emit_sequential(ctx, graph, grid_ids, bounds, trips, names, level + 1)
        scf_d.YieldOp([ctx.tensors.value(name) for name in names])
    for name, result in zip(names, for_op.results, strict=True):
        ctx.tensors.rebind(name, result)


def _bind_grid_iv(
    ctx: BuildContext, block_id: int, bound: LoopBounds, trip_iv: ir.Value
) -> None:
    ctx.block_id_to_trip_iv[block_id] = trip_iv
    ctx.bind_loop(
        block_id,
        _tile_offset(trip_iv, bound.begin, bound.step),
        (bound.begin, bound.end),
    )


def _tile_offset(trip_iv: ir.Value, begin: int, step: int) -> ir.Value:
    """Absolute tile offset ``begin + trip_iv * step`` for a normalized loop IV."""
    if begin == 0 and step == 1:
        return trip_iv
    expr = ir.AffineExpr.get_add(
        ir.AffineExpr.get_mul(ir.AffineDimExpr.get(0), ir.AffineConstantExpr.get(step)),
        ir.AffineConstantExpr.get(begin),
    )
    return affine_d.AffineApplyOp(ir.AffineMap.get(1, 0, [expr]), [trip_iv]).result


def _owned_region(
    ctx: BuildContext, tensor: ir.Value, owned: dict[int, OwnedDim]
) -> tuple[list[ir.Value], list[emit.Size]]:
    """Offsets and sizes of one iteration's region of ``tensor``: the real part of
    its tile along owned dims, everything elsewhere."""
    offsets: list[ir.Value] = []
    sizes: list[emit.Size] = []
    for dim, extent in enumerate(ir.RankedTensorType(tensor.type).shape):
        owner = owned.get(dim)
        if owner is None:
            offsets.append(ctx.index_const(0))
            sizes.append(extent)
        elif owner.point:
            offsets.append(ctx.block_id_to_iv[owner.block_id])
            sizes.append(1)
        else:
            offset, size, _ = tile_window(ctx, owner.block_id, 0, extent)
            offsets.append(offset)
            sizes.append(size)
    return offsets, sizes


@lowers(tracing_ops._for_loop, tracing_ops._for_loop_step)
def lower_nested_for_loop(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower a (possibly multi-dimensional) nested ``_for_loop``/``_for_loop_step``
    to one ``scf.for`` per block id.

    Block ids come from the loop's ``ForLoopGraphInfo``; bounds and steps from
    the node itself (``_for_loop(graph_id, begin, end, args[, step])``).
    """
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

    state_names = list(ctx.effects.writes(node.args[0]))
    return _emit_for_loop_level(
        ctx, node, body_graph, block_ids, loop_bounds, iter_arg_nodes, state_names, 0
    )


def _resolve_loop_bound(
    ctx: BuildContext,
    node: torch.fx.Node,
    source: object,
) -> int | ir.Value:
    """Resolve a ``_for_loop`` begin/end to a static int or an index value."""
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


def _lower_innermost_loop_body(
    ctx: BuildContext,
    body_graph: torch.fx.Graph,
    body_block: ir.Block,
    out_args: list,
    invariant_pairs: list[tuple],
    carried_pairs: list[tuple],
    iter_pairs: list[tuple],
) -> list[ir.Value]:
    """Bind placeholders, lower ``body_graph``, and collect Helion's carried values."""
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
    ctx.lower_graph(body_graph)
    yield_vals = []
    for a in out_args:
        v = ctx.get_value(a) if isinstance(a, torch.fx.Node) else None
        if v is not None:
            yield_vals.append(v)
    return yield_vals


def _emit_for_loop_level(
    ctx: BuildContext,
    node: torch.fx.Node,
    body_graph: torch.fx.Graph,
    block_ids: list[int],
    loop_bounds: list[LoopBounds],
    iter_arg_nodes: list,
    state_names: list[str],
    level: int,
) -> ir.Value:
    """Emit one ``scf.for`` for ``block_ids[level]``.

    Only the innermost level lowers ``body_graph``; outer levels of a combined
    multi-dim ``_for_loop`` node recurse. Iter args are Helion's carried values
    followed by the tensors written in the body.
    """
    block_id = block_ids[level]
    bounds = loop_bounds[level]
    is_innermost = level == len(block_ids) - 1
    begin = _resolve_loop_bound(ctx, node, bounds.begin)
    end = _resolve_loop_bound(ctx, node, bounds.end)

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
    else:
        out_args, iter_pairs, invariant_pairs, carried_pairs = [], [], [], []
    carried_init = [v for _, v in carried_pairs]

    for_op = scf_d.ForOp(
        ctx.as_index(begin),
        ctx.as_index(end),
        ctx.index_const(bounds.step),
        iter_args=[*carried_init, *(ctx.tensors.value(n) for n in state_names)],
    )
    body_block = for_op.body
    state_args = list(body_block.arguments[1 + len(carried_init) :])
    with (
        ir.InsertionPoint(body_block),
        ctx.enter_for_loop(block_id, body_block.arguments[0], (begin, end)),
    ):
        for name, arg in zip(state_names, state_args, strict=True):
            ctx.tensors.rebind(name, arg)
        if is_innermost:
            yield_vals = _lower_innermost_loop_body(
                ctx,
                body_graph,
                body_block,
                out_args,
                invariant_pairs,
                carried_pairs,
                iter_pairs,
            )
        else:
            _emit_for_loop_level(
                ctx,
                node,
                body_graph,
                block_ids,
                loop_bounds,
                iter_arg_nodes,
                state_names,
                level + 1,
            )
            yield_vals = []
        if len(yield_vals) > len(carried_init):
            raise NodeLoweringError(
                node,
                reason=(
                    "Loop body yielded more values than iter_args: "
                    f"{len(yield_vals)} > {len(carried_init)}"
                ),
                recovery_hint="Ensure loop-carried values match loop iter_args",
            )
        passthrough = body_block.arguments[1 + len(yield_vals) : 1 + len(carried_init)]
        scf_d.YieldOp(
            [
                *yield_vals,
                *passthrough,
                *(ctx.tensors.value(name) for name in state_names),
            ]
        )
    for name, result in zip(
        state_names, list(for_op.results)[len(carried_init) :], strict=True
    ):
        ctx.tensors.rebind(name, result)
    return for_op
