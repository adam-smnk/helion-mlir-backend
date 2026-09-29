"""Lowering of ``call_method`` FX nodes (``Tensor`` methods)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..support import UnsupportedOperationError
from ..support import torch_dtype_to_mlir
from . import emit
from .transpose_ops import lower_transpose
from .view_ops import view

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_call_method(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    method = node.target
    base = ctx.get_value(node.args[0]) if node.args else None
    if base is None:
        raise UnsupportedOperationError(
            f"Tensor.{method}", reason="the receiver has no lowered value"
        )

    if method in ("contiguous", "clone", "detach"):
        return base
    if method == "to":
        dtype = next(
            (arg for arg in node.args[1:] if isinstance(arg, torch.dtype)),
            node.kwargs.get("dtype"),
        )
        if dtype is None:
            value = node.meta.get("val")
            dtype = value.dtype if isinstance(value, torch.Tensor) else None
        return (
            base
            if dtype is None
            else emit.cast_tensor(base, torch_dtype_to_mlir(dtype))
        )
    if method in ("t", "permute", "transpose"):
        return lower_transpose(ctx, node)
    if method in ("view", "reshape"):
        reshaped = view(ctx, node)
        if reshaped is None:
            raise UnsupportedOperationError(
                f"Tensor.{method}",
                reason="only static shapes, or adding or removing unit dims, are supported",
            )
        return reshaped
    raise UnsupportedOperationError(f"Tensor.{method}", reason="method not lowered")
