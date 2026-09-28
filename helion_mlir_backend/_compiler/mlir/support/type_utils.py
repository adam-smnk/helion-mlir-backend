"""Torch dtype ↔ MLIR type utilities."""

from __future__ import annotations

import mlir.ir as ir
import torch

# Mapping from torch dtype to MLIR type factory lambda (called inside an
# ir.Context).
_DTYPE_TO_MLIR: dict[torch.dtype, str] = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
}

_MLIR_TO_DTYPE: dict[str, torch.dtype] = {
    mlir_name: dtype for dtype, mlir_name in _DTYPE_TO_MLIR.items()
}


def mlir_dtype_to_torch(name: str, default: torch.dtype = torch.float32) -> torch.dtype:
    """Convert an MLIR scalar type name to a PyTorch dtype."""
    return _MLIR_TO_DTYPE.get(name, default)


def torch_dtype_to_mlir(dtype: torch.dtype) -> ir.Type:
    """Convert a :mod:`torch` dtype to an MLIR scalar type.

    Must be called while an ``mlir.ir.Context`` is active.
    """

    from .errors import UnsupportedOperationError

    name = _DTYPE_TO_MLIR.get(dtype)
    if name is None:
        reason = (
            "arith/linalg integers are signless and this backend lowers them with "
            "signed semantics"
            if dtype == torch.uint8
            else "no MLIR element type mapping"
        )
        raise UnsupportedOperationError(f"dtype {dtype}", reason=reason)

    # Use ir.Type.parse for simple construction without individual factory calls.
    return ir.Type.parse(name)


def torch_tensor_to_mlir_type(fake_tensor: torch.Tensor) -> ir.Type:
    """Convert a fake/concrete :class:`torch.Tensor` to an MLIR RankedTensorType.

    Dynamic dimensions (``torch.SymInt``) are mapped to ``?`` (dynamic extent).
    Concrete integer dimensions are used as-is.

    Must be called while an ``mlir.ir.Context`` is active.
    """

    elem_ty = torch_dtype_to_mlir(fake_tensor.dtype)
    shape: list[int] = []
    for dim in fake_tensor.shape:
        if isinstance(dim, torch.SymInt):
            shape.append(ir.ShapedType.get_dynamic_size())
        else:
            shape.append(int(dim))
    return ir.RankedTensorType.get(shape, elem_ty)
