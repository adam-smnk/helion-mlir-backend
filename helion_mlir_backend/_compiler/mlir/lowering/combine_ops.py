"""``hl.reduce`` and ``hl.associative_scan`` (also ``torch.cumsum``) with a
combine function, as a sequential ``scf.for`` along the dimension.

The combine graph is lowered once per step on whole slices, which is valid
because a combine function is elementwise in its two arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion.language import reduce_ops
from helion.language import scan_ops
from mlir.dialects import affine as affine_d
from mlir.dialects import scf as scf_d
import mlir.ir as ir

from ..support import UnsupportedOperationError
from . import emit
from .control_flow import lower_subgraph
from .registry import lowers
from .view_ops import put
from .view_ops import static_reshape
from .view_ops import take

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from ..build_context import BuildContext


@lowers(reduce_ops._reduce)
def lower_reduce(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``_reduce(combine_graph, x, dim, keep_dims, is_tuple_input)``."""
    graph_id, source, dim, keep_dims, is_tuple = _operands(node, "hl.reduce")
    if dim is None:
        raise UnsupportedOperationError("hl.reduce", reason="dim=None")
    value, dim, extent = _source(ctx, source, dim)
    combine = _combine(ctx, graph_id)
    loop = scf_d.ForOp(
        ctx.index_const(1),
        ctx.index_const(extent),
        ctx.index_const(1),
        iter_args=[take(ctx, value, dim, ctx.index_const(0))],
    )
    with ir.InsertionPoint(loop.body):
        item = take(ctx, value, dim, loop.induction_variable)
        scf_d.YieldOp([combine(loop.inner_iter_args[0], item)])
    result = loop.results[0]
    if keep_dims:
        shape = list(ir.RankedTensorType(value.type).shape)
        shape[dim] = 1
        result = static_reshape(result, shape)
    return result


@lowers(scan_ops._associative_scan)
def lower_scan(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``_associative_scan(combine_graph, x, dim, reverse, is_tuple_input)``."""
    graph_id, source, dim, reverse, _ = _operands(node, "hl.associative_scan")
    value, dim, extent = _source(ctx, source, dim)
    combine = _combine(ctx, graph_id)
    value_type = ir.RankedTensorType(value.type)
    first = ctx.index_const(extent - 1 if reverse else 0)
    head = take(ctx, value, dim, first)
    out = emit.empty(list(value_type.shape), value_type.element_type)
    loop = scf_d.ForOp(
        ctx.index_const(1),
        ctx.index_const(extent),
        ctx.index_const(1),
        iter_args=[head, put(ctx, out, head, dim, first)],
    )
    with ir.InsertionPoint(loop.body):
        step = loop.induction_variable
        if reverse:
            d0 = ir.AffineDimExpr.get(0)
            position = affine_d.AffineApplyOp(
                ir.AffineMap.get(1, 0, [ir.AffineConstantExpr.get(extent - 1) - d0]),
                [step],
            ).result
        else:
            position = step
        acc, partial = loop.inner_iter_args
        item = take(ctx, value, dim, position)
        acc = combine(item, acc) if reverse else combine(acc, item)
        scf_d.YieldOp([acc, put(ctx, partial, acc, dim, position)])
    return loop.results[1]


def _operands(node: torch.fx.Node, name: str) -> tuple:
    defaults = (None, None, None, False, False)
    graph_id, source, dim, flag, is_tuple = (*node.args, *defaults[len(node.args) :])
    if is_tuple:
        raise UnsupportedOperationError(name, reason="tuple inputs")
    return graph_id, source, dim, flag, is_tuple


def _source(
    ctx: BuildContext, source: torch.fx.Node, dim: int
) -> tuple[ir.Value, int, int]:
    value = ctx.get_value(source)
    shape = ir.RankedTensorType(value.type).shape
    dim %= len(shape)
    return value, dim, shape[dim]


def _combine(
    ctx: BuildContext, graph_id: int
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    graph = ctx.host_function.device_ir.graphs[graph_id].graph

    def combine(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        (result,) = lower_subgraph(ctx, graph, [lhs, rhs])
        return result

    return combine
