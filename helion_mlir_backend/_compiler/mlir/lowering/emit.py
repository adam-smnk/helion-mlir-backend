"""Small MLIR builders shared by the lowering modules (need an active context)."""

from __future__ import annotations

from mlir.dialects import arith as arith_d
from mlir.dialects import linalg as linalg_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from ..support.errors import UnsupportedOperationError


def zero_attr(element_type: ir.Type) -> ir.Attribute:
    if isinstance(element_type, ir.FloatType):
        return ir.FloatAttr.get(element_type, 0.0)
    return ir.IntegerAttr.get(element_type, 0)


def constant(element_type: ir.Type, value: float) -> ir.Value:
    if isinstance(element_type, ir.FloatType):
        attr = ir.FloatAttr.get(element_type, float(value))
    else:
        attr = ir.IntegerAttr.get(element_type, int(value))
    return arith_d.ConstantOp(element_type, attr).result


def empty(shape: list[int], element_type: ir.Type) -> ir.Value:
    return tensor_d.EmptyOp(shape, element_type).result


def filled(shape: list[int], element_type: ir.Type, value: float) -> ir.Value:
    return linalg_d.fill(
        constant(element_type, value), outs=[empty(shape, element_type)]
    )


def cast_scalar(value: ir.Value, target: ir.Type) -> ir.Value | None:
    """Convert a scalar to ``target`` with torch semantics, or ``None``.

    Integers are signed except ``i1`` (bool), which extends as unsigned; a cast to
    ``i1`` is ``!= 0``.
    """
    source = value.type
    if source == target:
        return value
    if _is_bool(target) and isinstance(source, (ir.IntegerType, ir.FloatType)):
        zero = constant(source, 0)
        if isinstance(source, ir.FloatType):
            return arith_d.CmpFOp(arith_d.CmpFPredicate.UNE, value, zero).result
        return arith_d.CmpIOp(arith_d.CmpIPredicate.ne, value, zero).result
    if isinstance(source, ir.IndexType):
        if isinstance(target, ir.IntegerType):
            return arith_d.IndexCastOp(target, value).result
        if isinstance(target, ir.FloatType):
            integer = arith_d.IndexCastOp(ir.IntegerType.get_signless(64), value)
            return arith_d.SIToFPOp(target, integer.result).result
        return None
    if isinstance(source, ir.IntegerType):
        unsigned = _is_bool(source)
        if isinstance(target, ir.IndexType):
            return arith_d.IndexCastOp(target, value).result
        if isinstance(target, ir.IntegerType):
            if source.width == target.width:
                return value
            if source.width < target.width:
                extend = arith_d.ExtUIOp if unsigned else arith_d.ExtSIOp
                return extend(target, value).result
            return arith_d.TruncIOp(target, value).result
        if isinstance(target, ir.FloatType):
            convert = arith_d.UIToFPOp if unsigned else arith_d.SIToFPOp
            return convert(target, value).result
        return None
    if isinstance(source, ir.FloatType):
        if isinstance(target, ir.FloatType):
            if source.width == target.width:
                return value
            if source.width < target.width:
                return arith_d.ExtFOp(target, value).result
            return arith_d.TruncFOp(target, value).result
        if isinstance(target, ir.IntegerType):
            return arith_d.FPToSIOp(target, value).result
    return None


def _is_bool(element_type: ir.Type) -> bool:
    return isinstance(element_type, ir.IntegerType) and element_type.width == 1


def cast_tensor(value: ir.Value, element_type: ir.Type) -> ir.Value:
    """Elementwise cast as an all-parallel ``linalg.generic`` (fusable, vectorizable)."""
    source_type = ir.RankedTensorType(value.type)
    if source_type.element_type == element_type:
        return value
    shape = list(source_type.shape)
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(shape)))
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    generic = linalg_d.GenericOp(
        [ir.RankedTensorType.get(shape, element_type)],
        [value],
        [empty(shape, element_type)],
        ir.ArrayAttr.get([identity, identity]),
        ir.ArrayAttr.get([parallel] * len(shape)),
    )
    body = generic.regions[0].blocks.append(source_type.element_type, element_type)
    with ir.InsertionPoint(body):
        converted = cast_scalar(body.arguments[0], element_type)
        if converted is None:
            raise UnsupportedOperationError(
                "cast",
                reason=f"no cast from {source_type.element_type} to {element_type}",
            )
        linalg_d.YieldOp([converted])
    return generic.result
