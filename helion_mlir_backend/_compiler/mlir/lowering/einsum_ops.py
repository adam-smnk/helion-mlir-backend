"""Direct lowering of a captured ``torch.einsum`` into ``linalg.contract``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

from ..support.einsum_spec import ContractSpec
from ..support.einsum_spec import EinsumNotContractible
from ..support.einsum_spec import build_contract_spec

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


def lower_einsum(
    ctx: BuildContext, node: torch.fx.Node, out: ir.Value | None = None
) -> ir.Value | None:
    """Emit ``linalg.contract`` for a ``helion_mlir::einsum`` node.

    ``out``, when given, is a loop-carried accumulator the contraction adds
    into (matching ``linalg.contract``'s ``+ C[H]`` term).
    """
    from mlir.dialects import arith as arith_d
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from ..support.errors import UnsupportedOperationError
    from .matmul_ops import resolve_contraction_operand

    equation, operand_nodes = _einsum_node_args(node)
    if equation is None:
        return None

    subscripts: list[str] = []
    values: list[ir.Value] = []
    inputs_text = equation.replace(" ", "").split("->")[0].split(",")
    if len(inputs_text) != len(operand_nodes):
        return None
    for subscript, operand_node in zip(inputs_text, operand_nodes, strict=True):
        value, transposed = resolve_contraction_operand(ctx, operand_node)
        if value is None:
            return None
        values.append(value)
        # `resolve_contraction_operand` may hand back the pre-transpose value;
        # fold that transposition into the subscript instead.
        if transposed and len(subscript) >= 2:
            subscript = subscript[:-2] + subscript[-1] + subscript[-2]
        subscripts.append(subscript)

    output = equation.replace(" ", "").split("->")[1] if "->" in equation else None
    rebuilt = ",".join(subscripts) + (f"->{output}" if output is not None else "")

    shapes = [list(ir.RankedTensorType(value.type).shape) for value in values]
    try:
        spec = build_contract_spec(rebuilt, shapes)
    except EinsumNotContractible as error:
        raise UnsupportedOperationError(
            f"einsum('{equation}')",
            reason=str(error),
            alternatives=["rewrite the contraction as torch.matmul/torch.bmm"],
        ) from error

    lhs, rhs = values
    element_type = ir.RankedTensorType(lhs.type).element_type
    if element_type != ir.RankedTensorType(rhs.type).element_type:
        return None

    if out is None:
        accumulator_type = element_type
        if isinstance(accumulator_type, ir.FloatType):
            zero_attr = ir.FloatAttr.get(accumulator_type, 0.0)
        elif isinstance(accumulator_type, ir.IntegerType):
            zero_attr = ir.IntegerAttr.get(accumulator_type, 0)
        else:
            return None
        empty = tensor_d.EmptyOp(spec.out_shape, accumulator_type).result
        zero = arith_d.ConstantOp(accumulator_type, zero_attr).result
        out = linalg_d.fill(zero, outs=[empty])
    elif list(ir.RankedTensorType(out.type).shape) != spec.out_shape:
        return None

    return linalg_d.contract(
        lhs,
        rhs,
        outs=[out],
        indexing_maps=_indexing_maps(spec),
    )


def _indexing_maps(spec: ContractSpec) -> list[ir.AffineMap]:
    import mlir.ir as ir

    total = len(spec.iteration_dims)
    return [
        ir.AffineMap.get(
            total, 0, [ir.AffineDimExpr.get(position) for position in positions]
        )
        for positions in (spec.lhs, spec.rhs, spec.out)
    ]


def _einsum_node_args(
    node: torch.fx.Node,
) -> tuple[str | None, list[torch.fx.Node]]:
    args = list(node.args)
    if len(args) != 2 or not isinstance(args[0], str):
        return None, []
    operands = args[1]
    if not isinstance(operands, (list, tuple)) or not all(
        isinstance(operand, torch.fx.Node) for operand in operands
    ):
        return None, []
    if len(operands) != 2:
        return None, []
    return args[0], list(operands)
