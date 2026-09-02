"""Resolve Helion host tensor nodes and simple aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


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
    """Materialize a static reshape-style alias when its shape differs.

    A host-side ``.view()``/``.reshape()`` written outside the tiled loop
    produces a ``_host_tensor`` node that resolves to the *base* parameter's
    SSA value, which still carries the base shape. Emit the shape change so
    downstream slices see the alias's real geometry instead of silently
    using the base type.
    """
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    try:
        base_type = ir.RankedTensorType(base_value.type)
        base_shape = [int(dim) for dim in base_type.shape]
        element_type = base_type.element_type
    except Exception:
        # Native MLIR binding raises for a non-ranked-tensor type; treat as
        # "not aliasable" rather than a specific error.
        return None

    alias_shape = ctx.shape_from_node_meta(alias_node)
    if alias_shape is None or base_shape == alias_shape:
        return base_value

    alias_shape = [int(dimension) for dimension in alias_shape]
    if any(dimension < 0 for dimension in base_shape + alias_shape):
        # Dynamic extents have no static reassociation; bail out.
        return None

    base_numel = 1
    for dimension in base_shape:
        base_numel *= dimension
    alias_numel = 1
    for dimension in alias_shape:
        alias_numel *= dimension
    if base_numel != alias_numel:
        return None

    result_type = ir.RankedTensorType.get(alias_shape, element_type)

    if len(alias_shape) == 1:
        reassociation = [list(range(len(base_shape)))]
        return tensor_d.CollapseShapeOp(result_type, base_value, reassociation).result

    # General static N-D -> M-D relayout. Collapse to 1-D first so a single
    # reassociation is always valid, then expand into the alias shape.
    flat_value = base_value
    if len(base_shape) != 1:
        flat_type = ir.RankedTensorType.get([base_numel], element_type)
        flat_value = tensor_d.CollapseShapeOp(
            flat_type, base_value, [list(range(len(base_shape)))]
        ).result
    return tensor_d.ExpandShapeOp(
        result_type,
        flat_value,
        [list(range(len(alias_shape)))],
        [],
        alias_shape,
    ).result
