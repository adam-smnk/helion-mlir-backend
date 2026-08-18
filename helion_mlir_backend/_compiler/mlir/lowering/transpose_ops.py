"""Lowering for tile transpose / permute operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..build_context import BuildContext


_TRANSPOSE_TARGETS = (
    "aten.permute",
    "aten.transpose",
    "aten.t.default",
    "permute.default",
    "transpose.int",
    "t.default",
)


def is_transpose_node(node: object) -> bool:
    """Return whether an FX node is a permute/transpose of a tile."""
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_method":
        return str(node.target) in ("t", "permute", "transpose")
    if node.op != "call_function":
        return False
    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    return any(
        name in target_name or overload_name == name for name in _TRANSPOSE_TARGETS
    )


def transpose_permutation(node: torch.fx.Node) -> list[int] | None:
    """Return the resolved permutation for a transpose-family node."""
    import torch

    value = node.meta.get("val")
    rank = value.ndim if isinstance(value, torch.Tensor) else None

    target_name = str(node.target)
    overload_name = getattr(node.target, "__name__", "")
    is_method = node.op == "call_method"

    if (
        "aten.permute" in target_name
        or overload_name == "permute.default"
        or (is_method and target_name == "permute")
    ):
        dims = node.args[1] if len(node.args) > 1 else None
        if is_method and len(node.args) > 2:
            dims = list(node.args[1:])
        if not isinstance(dims, (list, tuple)):
            return None
        return [int(dim) for dim in dims]

    if rank is None:
        return None

    if (
        "aten.t.default" in target_name
        or overload_name == "t.default"
        or (is_method and target_name == "t")
    ):
        if rank != 2:
            return None
        return [1, 0]

    if (
        "aten.transpose" in target_name
        or overload_name == "transpose.int"
        or (is_method and target_name == "transpose")
    ):
        if len(node.args) < 3:
            return None
        first = int(node.args[1]) % rank
        second = int(node.args[2]) % rank
        permutation = list(range(rank))
        permutation[first], permutation[second] = second, first
        return permutation

    return None


def swaps_last_two_dims(node: torch.fx.Node) -> bool:
    """Return whether the node only exchanges the two innermost dimensions."""
    permutation = transpose_permutation(node)
    if permutation is None:
        return False
    rank = len(permutation)
    if rank < 2:
        return False
    return permutation == [*range(rank - 2), rank - 1, rank - 2]


def lower_transpose(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower a standalone permute/transpose to ``linalg.transpose``."""
    from mlir.dialects import linalg as linalg_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    if not is_transpose_node(node) or not node.args:
        return None

    source = ctx.get_value(node.args[0])
    if source is None:
        return None

    permutation = transpose_permutation(node)
    if permutation is None:
        return None

    source_type = ir.RankedTensorType(source.type)
    source_shape = list(source_type.shape)
    if len(permutation) != len(source_shape):
        return None

    result_shape = [source_shape[dim] for dim in permutation]
    init = tensor_d.EmptyOp(result_shape, source_type.element_type).result
    operation = linalg_d.transpose(source, outs=[init], permutation=permutation)
    return operation.results[0]
