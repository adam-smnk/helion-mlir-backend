"""Torch dtype ↔ MLIR type utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import mlir.ir as ir


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
    torch.uint8: "ui8",
    torch.bool: "i1",
}


def torch_dtype_to_mlir(dtype: torch.dtype) -> ir.Type:
    """Convert a :mod:`torch` dtype to an MLIR scalar type.

    Must be called while an ``mlir.ir.Context`` is active.
    """
    import mlir.ir as ir

    name = _DTYPE_TO_MLIR.get(dtype)
    if name is None:
        raise NotImplementedError(f"No MLIR type mapping for torch dtype {dtype}")

    # Use ir.Type.parse for simple construction without individual factory calls.
    return ir.Type.parse(name)


def torch_tensor_to_mlir_type(fake_tensor: torch.Tensor) -> ir.Type:
    """Convert a fake/concrete :class:`torch.Tensor` to an MLIR RankedTensorType.

    Dynamic dimensions (``torch.SymInt``) are mapped to ``?`` (dynamic extent).
    Concrete integer dimensions are used as-is.

    Must be called while an ``mlir.ir.Context`` is active.
    """
    import mlir.ir as ir

    elem_ty = torch_dtype_to_mlir(fake_tensor.dtype)
    shape: list[int] = []
    for dim in fake_tensor.shape:
        if isinstance(dim, torch.SymInt):
            shape.append(ir.ShapedType.get_dynamic_size())
        else:
            shape.append(int(dim))
    return ir.RankedTensorType.get(shape, elem_ty)


def get_zero_attr(dtype: torch.dtype) -> ir.Attribute:
    """Return an MLIR attribute representing zero for *dtype*.

    Must be called while an ``mlir.ir.Context`` is active.
    """
    import mlir.ir as ir

    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        mlir_ty = torch_dtype_to_mlir(dtype)
        return ir.FloatAttr.get(mlir_ty, 0.0)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        mlir_ty = torch_dtype_to_mlir(dtype)
        return ir.IntegerAttr.get(mlir_ty, 0)
    if dtype == torch.bool:
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)
    raise NotImplementedError(f"No zero attr for dtype {dtype}")
