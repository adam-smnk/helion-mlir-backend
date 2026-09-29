"""Resolve Helion host tensor nodes and simple aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import sympy
import torch

from .registry import lowers

if TYPE_CHECKING:
    from ..build_context import BuildContext


@lowers(tracing_ops._host_tensor)
def lower_host_tensor(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower ``_host_tensor('name')`` to a function argument value."""
    name = node.args[0]
    assert isinstance(name, str)
    if name in ctx.param_to_value:
        return ctx.param_to_value[name]

    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        resolved = resolve_host_tensor_alias_value(ctx, value)
        if resolved is not None:
            aliased = materialize_host_tensor_alias_shape(ctx, resolved, node)
            return aliased if aliased is not None else resolved
    return None


def resolve_host_tensor_alias_value(
    ctx: BuildContext, tensor: torch.Tensor
) -> ir.Value | None:
    """Resolve a host tensor alias through its origin and base chain."""
    seen: set[int] = set()
    current: torch.Tensor | None = tensor
    while isinstance(current, torch.Tensor) and id(current) not in seen:
        seen.add(id(current))
        origin = ctx.host_function.tensor_to_origin.get(current)
        if origin is not None:
            host_name = origin.host_str()
            if host_name in ctx.param_to_value:
                return ctx.param_to_value[host_name]
        current = getattr(current, "_base", None)
    return None


def materialize_host_tensor_alias_shape(
    ctx: BuildContext,
    base_value: ir.Value,
    alias_node: torch.fx.Node,
) -> ir.Value | None:
    """Materialize a reshape-style alias when its shape differs.

    A host-side ``.view()``/``.reshape()`` written outside the tiled loop
    produces a ``_host_tensor`` node that resolves to the *base* parameter's
    SSA value, which still carries the base shape. Emit the shape change so
    downstream slices see the alias's real geometry instead of silently
    using the base type. Runtime sizes come from ``ctx.sizes``.
    """

    base_type = base_value.type
    if not isinstance(base_type, ir.RankedTensorType):
        return None
    element_type = base_type.element_type
    alias_sizes = [ctx.sizes.expr(size) for size in alias_node.meta["val"].shape]
    base_name = next(
        (name for name, value in ctx.param_to_value.items() if value == base_value),
        None,
    )
    if base_name is not None and alias_sizes == ctx.sizes.ref(base_name):
        return base_value
    dynamic = ir.ShapedType.get_dynamic_size()
    base_shape = list(base_type.shape)
    alias_shape = [dynamic if size.free_symbols else int(size) for size in alias_sizes]
    if base_shape == alias_shape and dynamic not in alias_shape:
        return base_value
    if base_name is not None:
        base_numel = sympy.prod(ctx.sizes.ref(base_name))
    elif dynamic not in base_shape:
        base_numel = sympy.prod([sympy.Integer(size) for size in base_shape])
    else:
        return None
    if sympy.prod(alias_sizes) != base_numel:
        if not base_numel.free_symbols:
            return None
        from ..support import UnsupportedOperationError

        raise UnsupportedOperationError(
            f"host view '{alias_node.args[0]}'",
            reason=f"cannot show that {alias_sizes} has {base_numel} elements",
        )

    result_type = ir.RankedTensorType.get(alias_shape, element_type)

    if len(alias_shape) == 1:
        reassociation = [list(range(len(base_shape)))]
        return tensor_d.CollapseShapeOp(result_type, base_value, reassociation).result

    # General N-D -> M-D relayout. Collapse to 1-D first so a single
    # reassociation is always valid, then expand into the alias shape.
    flat_value = base_value
    if len(base_shape) != 1:
        flat_dim = dynamic if base_numel.free_symbols else int(base_numel)
        flat_type = ir.RankedTensorType.get([flat_dim], element_type)
        flat_value = tensor_d.CollapseShapeOp(
            flat_type, base_value, [list(range(len(base_shape)))]
        ).result
    output = [ctx.sizes.value(size) for size in alias_sizes]
    return tensor_d.ExpandShapeOp(
        result_type,
        flat_value,
        [list(range(len(alias_shape)))],
        [size for size in output if isinstance(size, ir.Value)],
        alias_shape,
    ).result
