"""Lower tensor creation operations to Linalg-on-Tensors IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..support.errors import DynamicShapeError
from ..support.errors import NodeLoweringError
from ..support.errors import ShapeError
from ..support.type_utils import get_zero_attr
from ..support.type_utils import torch_dtype_to_mlir

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_full(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower ``full(shape, fill_val, dtype)`` to an initialized tensor."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    try:
        shape_nodes = node.args[0]
        fill_value = node.args[1]
        dtype = node.args[2] if len(node.args) > 2 else torch.float32
        shape = ctx.shape_from_nodes(shape_nodes, "full")
        mlir_dtype = torch_dtype_to_mlir(dtype)
        empty = tensor_d.EmptyOp(shape, mlir_dtype).result
        fill_attr = (
            ir.FloatAttr.get(mlir_dtype, float(fill_value))
            if isinstance(mlir_dtype, ir.FloatType)
            else ir.IntegerAttr.get(mlir_dtype, int(fill_value))
        )
        fill_constant = arith_d.ConstantOp(mlir_dtype, fill_attr).result
        return linalg_d.fill(fill_constant, outs=[empty])
    except (ShapeError, DynamicShapeError):
        raise
    except Exception as exc:
        raise NodeLoweringError(
            node,
            reason=str(exc),
            recovery_hint="Check tensor shapes and dtypes",
        ) from exc


def lower_zeros(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower ``zeros(shape, dtype)`` to a zero-initialized tensor."""
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d

    try:
        shape_nodes = node.args[0]
        dtype = node.args[1] if len(node.args) > 1 else torch.float32
        shape = ctx.shape_from_nodes(shape_nodes, "zeros")
        mlir_dtype = torch_dtype_to_mlir(dtype)
        empty = tensor_d.EmptyOp(shape, mlir_dtype).result
        zero = arith_d.ConstantOp(mlir_dtype, get_zero_attr(dtype)).result
        return linalg_d.fill(zero, outs=[empty])
    except (ShapeError, DynamicShapeError):
        raise
    except Exception as exc:
        raise NodeLoweringError(
            node,
            reason=str(exc),
            recovery_hint="Check tensor shapes and dtypes",
        ) from exc
