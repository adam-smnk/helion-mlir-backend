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
    """Materialize a static flatten-style alias when its shape differs."""
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

    if len(alias_shape) != 1:
        return None
    base_numel = 1
    for dimension in base_shape:
        base_numel *= dimension
    if base_numel != int(alias_shape[0]):
        return None

    result_type = ir.RankedTensorType.get(alias_shape, element_type)
    reassociation = [list(range(len(base_shape)))]
    return tensor_d.CollapseShapeOp(result_type, base_value, reassociation).result
