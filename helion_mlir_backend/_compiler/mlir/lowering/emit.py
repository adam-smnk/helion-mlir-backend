"""Small MLIR builders shared by the lowering modules (need an active context)."""

from __future__ import annotations

from dataclasses import dataclass

from mlir.dialects import affine as affine_d
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


Size = int | ir.Value
"""A static size, or an ``index`` value for a dynamic one."""


@dataclass
class Results:
    """Several values of one node, read by ``getitem``."""

    results: list[ir.Value]


def affine_min(results: list[ir.AffineExpr], operands: list[ir.Value]) -> ir.Value:
    """``affine.min`` of ``results`` over dims ``d0..`` bound to ``operands``."""
    return affine_d.AffineMinOp(
        ir.AffineMap.get(len(operands), 0, results), operands
    ).result


def _mixed(sizes: list[Size]) -> tuple[list[ir.Value], list[int]]:
    dynamic = ir.ShapedType.get_dynamic_size()
    return (
        [size for size in sizes if isinstance(size, ir.Value)],
        [dynamic if isinstance(size, ir.Value) else size for size in sizes],
    )


def extract_slice(
    tensor: ir.Value, offsets: list[ir.Value], sizes: list[Size]
) -> ir.Value:
    """Unit-stride ``extract_slice`` without rank reduction."""
    dynamic_sizes, static_sizes = _mixed(sizes)
    element_type = ir.RankedTensorType(tensor.type).element_type
    return tensor_d.ExtractSliceOp(
        ir.RankedTensorType.get(static_sizes, element_type),
        tensor,
        offsets,
        dynamic_sizes,
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
        static_sizes=static_sizes,
        static_strides=[1] * len(offsets),
    ).result


def insert_slice(
    value: ir.Value, dest: ir.Value, offsets: list[ir.Value], sizes: list[Size]
) -> ir.Value:
    """Unit-stride ``insert_slice`` (``value`` may drop unit dims of the slice)."""
    dynamic_sizes, static_sizes = _mixed(sizes)
    return tensor_d.InsertSliceOp(
        value,
        dest,
        offsets,
        dynamic_sizes,
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
        static_sizes=static_sizes,
        static_strides=[1] * len(offsets),
    ).result


def parallel_insert_slice(
    value: ir.Value, dest: ir.Value, offsets: list[ir.Value], sizes: list[Size]
) -> None:
    dynamic_sizes, static_sizes = _mixed(sizes)
    tensor_d.ParallelInsertSliceOp(
        value,
        dest,
        offsets,
        dynamic_sizes,
        [],
        static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
        static_sizes=static_sizes,
        static_strides=[1] * len(offsets),
    )


def pad_high(value: ir.Value, sizes: list[Size], shape: list[int]) -> ir.Value:
    """Zero-pad ``value`` (of ``sizes``) at the end of each dimension to ``shape``."""
    d0 = ir.AffineDimExpr.get(0)
    highs: list[Size] = [
        target - size
        if isinstance(size, int)
        else affine_d.AffineApplyOp(
            ir.AffineMap.get(1, 0, [ir.AffineConstantExpr.get(target) - d0]),
            [size],
        ).result
        for size, target in zip(sizes, shape, strict=True)
    ]
    dynamic_highs, static_highs = _mixed(highs)
    element_type = ir.RankedTensorType(value.type).element_type
    pad = tensor_d.PadOp(
        ir.RankedTensorType.get(shape, element_type),
        value,
        [],
        dynamic_highs,
        [0] * len(shape),
        static_highs,
    )
    body = pad.regions[0].blocks.append(*[ir.IndexType.get()] * len(shape))
    with ir.InsertionPoint(body):
        tensor_d.YieldOp(constant(element_type, 0))
    return pad.result


def mask(value: ir.Value, bounds: dict[int, ir.Value], other: float) -> ir.Value:
    """``value`` where every ``index(dim) < bounds[dim]``, else ``other``."""
    value_type = ir.RankedTensorType(value.type)
    shape, element_type = list(value_type.shape), value_type.element_type
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(shape)))
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    generic = linalg_d.GenericOp(
        [value_type],
        [value],
        [empty(shape, element_type)],
        ir.ArrayAttr.get([identity, identity]),
        ir.ArrayAttr.get([parallel] * len(shape)),
    )
    body = generic.regions[0].blocks.append(element_type, element_type)
    with ir.InsertionPoint(body):
        inside = None
        for dim, bound in bounds.items():
            below = arith_d.CmpIOp(
                arith_d.CmpIPredicate.ult, linalg_d.IndexOp(dim).result, bound
            ).result
            inside = below if inside is None else arith_d.AndIOp(inside, below).result
        linalg_d.YieldOp(
            [arith_d.SelectOp(inside, body.arguments[0], constant(element_type, other))]
        )
    return generic.result
