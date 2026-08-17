"""Registry for ATen operations with custom MLIR lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from .build_context import BuildContext


_CUSTOM_TARGETS = (
    "aten.addmm",
    "aten.mm",
    "aten.matmul",
    "aten.bmm",
    "aten.baddbmm",
)


def aten_target_matches(node: torch.fx.Node, *names: str) -> bool:
    """Match ATen targets by canonical name or FX overload short name."""
    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    return any(name in target_name or overload_name == name for name in names)


def is_custom_aten(node: torch.fx.Node) -> bool:
    """Return whether an ATen node is reserved for custom lowering."""
    return node.op == "call_function" and aten_target_matches(node, *_CUSTOM_TARGETS)


def lower_custom_aten(builder: object, node: torch.fx.Node) -> ir.Value | None:
    """Try the registered custom ATen lowerers in precedence order."""
    if aten_target_matches(node, "aten.addmm", "addmm.default"):
        lowered = lower_addmm(builder.context, node)
        if lowered is not None:
            return lowered

    if aten_target_matches(node, "aten.baddbmm", "baddbmm.default"):
        lowered = builder._lower_aten_baddbmm(node)
        if lowered is not None:
            return lowered

    if aten_target_matches(node, "aten.add.Tensor", "add.Tensor"):
        lowered = lower_add_matmul_accumulate(builder.context, node)
        if lowered is not None:
            return lowered
        lowered = builder._lower_aten_add_tensor(node)
        if lowered is not None:
            return lowered

    if aten_target_matches(
        node,
        "aten.mm",
        "aten.matmul",
        "aten.bmm",
        "mm.default",
        "matmul.default",
        "bmm.default",
    ):
        lowered = builder._lower_aten_matmul(node)
        if lowered is not None:
            return lowered

    lowered = lower_passthrough(builder.context, node)
    if lowered is not None:
        return lowered

    return None


def lower_passthrough(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower shape-preserving ATen aliases without emitting a helper call."""
    from .aten_lowering import normalized_aten_args

    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    if not any(
        operation in target_name
        for operation in ("aten.alias", "aten.detach", "aten.clone", "aten.contiguous")
    ) and overload_name not in {
        "alias.default",
        "detach.default",
        "clone.default",
        "contiguous.default",
    }:
        return None
    args = list(normalized_aten_args(node))
    if args and isinstance(args[0], torch.fx.Node):
        return ctx.get_value(args[0])
    return None


def lower_addmm(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower unit-scaled ``aten.addmm`` into an accumulator matmul."""
    from mlir.dialects import linalg as linalg_d

    from .aten_lowering import normalized_aten_args

    args = list(normalized_aten_args(node))
    if len(args) < 3:
        return None
    accumulator = ctx.get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
    lhs = ctx.get_value(args[1]) if isinstance(args[1], torch.fx.Node) else None
    rhs = ctx.get_value(args[2]) if isinstance(args[2], torch.fx.Node) else None
    if accumulator is None or lhs is None or rhs is None:
        return None
    beta = args[3] if len(args) > 3 else 1
    alpha = args[4] if len(args) > 4 else 1
    if beta != 1 or alpha != 1:
        return None
    return linalg_d.matmul(lhs, rhs, outs=[accumulator])


def lower_add_matmul_accumulate(
    ctx: BuildContext, node: torch.fx.Node
) -> ir.Value | None:
    """Lower ``acc + matmul`` into a loop-carried accumulator update."""
    from .aten_lowering import normalized_aten_args
    from .matmul_ops import emit_matmul_like

    args = list(normalized_aten_args(node))
    if len(args) < 2 or (args[2] if len(args) > 2 else 1) != 1:
        return None
    accumulator_node: torch.fx.Node | None = None
    matmul_node: torch.fx.Node | None = None
    for first, second in ((args[0], args[1]), (args[1], args[0])):
        if not isinstance(first, torch.fx.Node) or not isinstance(
            second, torch.fx.Node
        ):
            continue
        if aten_target_matches(
            second,
            "aten.mm",
            "aten.matmul",
            "aten.bmm",
            "mm.default",
            "matmul.default",
            "bmm.default",
        ):
            accumulator_node = first
            matmul_node = second
            break
    if accumulator_node is None or matmul_node is None:
        return None
    accumulator = ctx.get_value(accumulator_node)
    matmul_args = list(normalized_aten_args(matmul_node))
    if accumulator is None or len(matmul_args) < 2:
        return None
    lhs = (
        ctx.get_value(matmul_args[0])
        if isinstance(matmul_args[0], torch.fx.Node)
        else None
    )
    rhs = (
        ctx.get_value(matmul_args[1])
        if isinstance(matmul_args[1], torch.fx.Node)
        else None
    )
    if lhs is None or rhs is None:
        return None
    return emit_matmul_like(ctx, lhs, rhs, out=accumulator)
