"""Call-site lowering of a generic ATen node through its torch-mlir helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlir.dialects import func as func_d
import torch.fx

from ..support import NodeLoweringError
from ..support import UnsupportedOperationError

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_via_aten_helper(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Emit ``func.call`` to the node's pre-built helper, rebuilding it if needed."""

    from ..aten_lowering import collect_tensor_input_positions
    from ..aten_lowering import normalized_aten_args
    from .helper_rebuild import rebuild_aten_helper_for_call

    helpers = ctx.aten_helpers
    entry = helpers.get(id(node)) if helpers is not None else None
    if entry is None:
        raise UnsupportedOperationError(
            str(node.target),
            reason="no torch-mlir helper was built for this ATen node "
            "(see the preprocessing warnings above)",
        )
    func_name, return_types = entry
    args = normalized_aten_args(node)
    operands = [
        ctx.get_value(args[position])
        for position in collect_tensor_input_positions(node)
        if isinstance(args[position], torch.fx.Node)
        and ctx.get_value(args[position]) is not None
    ]

    if not helpers.signature_matches(func_name, operands):
        # The pre-built helper came from a "typical" tile shape; rebuild it for
        # the operand types at this call site (e.g. a boundary tile).
        rebuilt = rebuild_aten_helper_for_call(ctx, node, operands)
        if rebuilt is not None:
            func_name, return_types = rebuilt
    if not helpers.signature_matches(func_name, operands):
        raise NodeLoweringError(
            node,
            reason=(
                f"ATen helper '{func_name}' signature does not match operand types "
                f"{[str(value.type) for value in operands]} even after rebuilding"
            ),
        )

    call = func_d.CallOp(return_types, func_name, operands)
    return call.results[0] if call.results else None
