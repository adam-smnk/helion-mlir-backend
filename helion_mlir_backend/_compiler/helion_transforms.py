"""Python-defined transform ops of the opt pipeline and the schedules applying them.

The ops share a module because a Python-defined dialect only registers the ops
defined before it is loaded.

``pin_transposes``: lighthouse's register tiling unrolls a transpose with a
narrow inner dim into per-element pieces, read from a temporary of its padded
source tile. A small transpose that only moves a loaded tile is pinned untiled,
so it becomes one vector transpose.

``vectorize_pads``: bufferized, a pad is a temporary zeroed and then copied
into, and vectorize_all's copy forwarding (upstream
``LinalgCopyVTRForwardingPattern``) reads past the copy with poison padding. As
a vector read of the pad's source with the pad value past its end, written into
an empty tensor, there is no such copy.

``split_transfers``: tile extents are runtime values, so every tile's transfer
may be out of bounds and lowers to masked accesses. Each is split on an
in-bounds check, so only edge tiles take the masked path. Upstream's
``vector.split_transfer_full_partial`` stages n-D vectors through a stack buffer
with ``vector.type_cast``, which overflows it when the inner vector dim is not a
power of two in bytes, and loops forever on rank-reducing transfers.

``legalize_for_llvm``: the vector ops upstream's LLVM lowering rejects or gets
wrong become per-element scalar loads and stores: 0-d transfers (lowered only on
memrefs of unit inner stride) and i1 transfers (LLVM packs i1 vectors into bits,
while a memref holds one byte per i1). Contractions with operands narrower than
the accumulator (folded extensions, for x86 dot-product and AMX patterns) that
no x86 pattern took get their operands extended again.

``vectorize_linalg``: lighthouse vectorizes without vector sizes, which fails
for ops of runtime shape. Tiling bounds each runtime extent (an ``affine.min``
with a constant), so such ops are vectorized with masks, the bounds as vector
sizes. Masked contractions are unmasked for upstream's x86 contraction patterns.
"""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
import itertools
import math

from lighthouse.dialects import DialectExtension
from lighthouse.dialects.transform.transform_ext.utils.tile_size_analysis import (
    TILE_SIZES_ATTR_NAME,
)
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import affine
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector as transform_vector


class HelionTransformDialect(DialectExtension, name="helion_transform"):
    """Transform ops of the Helion MLIR backend's pipelines."""


def _transform_op(*, modifies_payload: bool) -> Callable[[type], type]:
    """Attach the interfaces of a transform op applied by its static ``run``.

    An op modifying the payload produces no handles; others only read it.
    """

    def decorate(cls: type) -> type:
        class Transform(transform.TransformOpInterface):
            @staticmethod
            def apply(
                op: ir.OpView,
                rewriter: transform.TransformRewriter,
                results: transform.TransformResults,
                state: transform.TransformState,
            ) -> DiagnosedSilenceableFailure:
                return cls.run(op, rewriter, results, state)

            @staticmethod
            def allow_repeated_handle_operands(_op: ir.OpView) -> bool:
                return False

        class Effects(ir.MemoryEffectsOpInterface):
            @staticmethod
            def get_effects(op: ir.OpView) -> list:
                effects = transform.only_reads_handle(op.op_operands)
                if modifies_payload:
                    return effects + transform.modifies_payload()
                return (
                    effects
                    + transform.produces_handle(op.results)
                    + transform.only_reads_payload()
                )

        def attach_interface_impls(context: ir.Context | None = None) -> None:
            Transform.attach(cls.OPERATION_NAME, context=context)
            Effects.attach(cls.OPERATION_NAME, context=context)

        cls.attach_interface_impls = staticmethod(attach_interface_impls)
        return cls

    return decorate


def _payload_ops(
    state: transform.TransformState, handle: ir.Value, op_types: type | tuple
) -> list[ir.OpView]:
    """The ``op_types`` ops nested in the payload of ``handle``, in pre-order."""
    found: list[ir.OpView] = []

    def collect(visited: ir.Operation) -> ir.WalkResult:
        if isinstance(visited.opview, op_types):
            found.append(visited.opview)
        return ir.WalkResult.ADVANCE

    for target in state.get_payload_ops(handle):
        target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
    return found


def _pad_value(pad: tensor.PadOp) -> ir.Value | None:
    """The pad's constant padding value, usable before the pad, if it has one."""
    body = pad.region.blocks[0]
    value = list(body.operations)[-1].operands[0]
    if isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, arith.ConstantOp
    ):
        return arith.ConstantOp(value.type, value.owner.opview.value).result
    if isinstance(value, ir.BlockArgument) and value.owner == body:
        return None
    if isinstance(value, ir.OpResult) and value.owner.block == body:
        return None
    return value


def _vectorize(pad: tensor.PadOp, rewriter: transform.TransformRewriter) -> None:
    result_type = ir.RankedTensorType(pad.result.type)
    if list(pad.low) or any(pad.static_low):
        return
    rank = result_type.rank
    # A runtime extent is read up to its bound, masked past the tensor's end.
    shape = [_dim_bound(pad.result, dim) for dim in range(rank)]
    if None in shape or (
        not result_type.has_static_shape and math.prod(shape) > _MAX_VECTOR_ELEMENTS
    ):
        return
    with ir.InsertionPoint(pad), pad.location:
        padding = _pad_value(pad)
        if padding is None:
            return
        source_shape = ir.RankedTensorType(pad.source.type).shape
        identity = ir.AffineMap.get_minor_identity(rank, rank)
        index = ir.IndexType.get()
        origin = [arith.ConstantOp(index, 0).result] * rank
        read = vector.TransferReadOp(
            ir.VectorType.get(shape, result_type.element_type),
            pad.source,
            origin,
            identity,
            padding,
            [real == padded for real, padded in zip(source_shape, shape, strict=True)],
        ).result
        sizes: list[int | ir.Value] = list(result_type.shape)
        for dim in range(rank):
            if result_type.is_dynamic_dim(dim):
                size = tensor.DimOp(pad.source, arith.ConstantOp(index, dim)).result
                high = _pad_amount(pad, True, dim)
                if high != 0:
                    if isinstance(high, int):
                        high = arith.ConstantOp(index, high).result
                    size = arith.AddIOp(size, high).result
                sizes[dim] = size
        empty = tensor.EmptyOp(sizes, result_type.element_type).result
        written = vector.TransferWriteOp(
            result_type,
            read,
            empty,
            origin,
            identity,
            [not result_type.is_dynamic_dim(dim) for dim in range(rank)],
        ).result
    rewriter.replace_op(pad, [written])


@_transform_op(modifies_payload=True)
class VectorizePadsOp(HelionTransformDialect.Operation, name="vectorize_pads"):
    """Vectorize every ``tensor.pad`` in the target with no low padding, a
    constant padding value and evident bounds of its runtime extents."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "VectorizePadsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for pad in _payload_ops(state, op.target, tensor.PadOp):
            _vectorize(pad, rewriter)
        return DiagnosedSilenceableFailure.Success


@_transform_op(modifies_payload=True)
class MaterializeCopiesOp(HelionTransformDialect.Operation, name="materialize_copies"):
    """Insert a ``linalg.copy`` into the destination slice of every insert in the
    target of a static slice of more than ``_MAX_VECTOR_ELEMENTS`` elements."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MaterializeCopiesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        insert_types = (tensor.InsertSliceOp, tensor.ParallelInsertSliceOp)
        for insert in _payload_ops(state, op.target, insert_types):
            _materialize_copy(insert)
        return DiagnosedSilenceableFailure.Success


def _materialize_copy(insert: tensor.InsertSliceOp) -> None:
    if _is_empty(insert.dest):
        return
    source = _moved_slice(insert.source)
    if source is None:
        return
    source_type = ir.RankedTensorType(source.type)
    sizes = list(insert.static_sizes)
    if (
        not source_type.has_static_shape
        or math.prod(source_type.shape) <= _MAX_VECTOR_ELEMENTS
        or [size for size in sizes if size != 1]
        != [dim for dim in source_type.shape if dim != 1]
    ):
        return
    # Ops of an scf.forall's in_parallel region go before it.
    anchor = insert
    if isinstance(insert, tensor.ParallelInsertSliceOp):
        anchor = insert.operation.parent
    with ir.InsertionPoint(anchor), insert.location:
        destination = tensor.ExtractSliceOp(
            source_type,
            insert.dest,
            insert.offsets,
            insert.sizes,
            insert.strides,
            static_offsets=insert.static_offsets,
            static_sizes=insert.static_sizes,
            static_strides=insert.static_strides,
        ).result
        copied = linalg.copy(source, outs=[destination])
    insert.operation.operands[0] = copied


def _moved_slice(value: ir.Value) -> ir.Value | None:
    """The slice of another tensor ``value`` is, looking through inserts that fill
    a whole empty tensor (unit dims added), else ``None``."""
    if not isinstance(value, ir.OpResult):
        return None
    op = value.owner.opview
    if isinstance(op, tensor.ExtractSliceOp):
        return value
    if (
        isinstance(op, tensor.InsertSliceOp)
        and _is_empty(op.dest)
        and list(op.static_sizes) == list(ir.RankedTensorType(op.dest.type).shape)
    ):
        return _moved_slice(op.source)
    return None


def _is_empty(value: ir.Value) -> bool:
    return isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, tensor.EmptyOp
    )


def _split(
    transfer: vector.TransferReadOp | vector.TransferWriteOp,
    rewriter: transform.TransformRewriter,
) -> None:
    base_type = transfer.base.type
    if not isinstance(base_type, ir.MemRefType) or transfer.mask is not None:
        return
    is_read = isinstance(transfer, vector.TransferReadOp)
    vector_type = ir.VectorType(
        transfer.result.type if is_read else transfer.valueToStore.type
    )
    rank, vector_rank = base_type.rank, vector_type.rank
    in_bounds = [ir.BoolAttr(flag).value for flag in transfer.in_bounds]
    minor_identity = ir.AffineMap.get_minor_identity(rank, vector_rank)
    if all(in_bounds) or transfer.permutation_map.value != minor_identity:
        return
    index = ir.IndexType.get()
    indices = list(transfer.indices)
    with ir.InsertionPoint(transfer), transfer.location:
        fits = None
        for dim, (size, flag) in enumerate(
            zip(vector_type.shape, in_bounds, strict=True), rank - vector_rank
        ):
            if flag:
                continue
            end = arith.AddIOp(indices[dim], arith.ConstantOp(index, size)).result
            extent = memref.DimOp(transfer.base, arith.ConstantOp(index, dim)).result
            dim_fits = arith.CmpIOp(arith.CmpIPredicate.sle, end, extent).result
            fits = dim_fits if fits is None else arith.AndIOp(fits, dim_fits).result
        branch = scf.IfOp(fits, [vector_type] if is_read else [], has_else=True)
        for block, flags in (
            (branch.then_block, [True] * vector_rank),
            (branch.else_block, in_bounds),
        ):
            with ir.InsertionPoint(block):
                if is_read:
                    read = vector.TransferReadOp(
                        vector_type,
                        transfer.base,
                        indices,
                        minor_identity,
                        transfer.padding,
                        flags,
                    ).result
                    scf.YieldOp([read])
                else:
                    vector.TransferWriteOp(
                        None,
                        transfer.valueToStore,
                        transfer.base,
                        indices,
                        minor_identity,
                        flags,
                    )
                    scf.YieldOp([])
    if is_read:
        rewriter.replace_op(transfer, list(branch.results))
    else:
        rewriter.erase_op(transfer)


@_transform_op(modifies_payload=True)
class SplitTransfersOp(HelionTransformDialect.Operation, name="split_transfers"):
    """Guard every possibly out-of-bounds, minor-identity, unmasked memref transfer
    in the target with an in-bounds check: an in-bounds transfer if it passes, the
    original otherwise."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "SplitTransfersOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        transfer_types = (vector.TransferReadOp, vector.TransferWriteOp)
        for transfer in _payload_ops(state, op.target, transfer_types):
            _split(transfer, rewriter)
        return DiagnosedSilenceableFailure.Success


@_transform_op(modifies_payload=True)
class PinTransposesOp(HelionTransformDialect.Operation, name="pin_transposes"):
    """Annotate every ``_is_small_transpose`` op in the target with zero tile
    sizes: lighthouse's tiling keeps annotated sizes, so it stays one vector
    transpose."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "PinTransposesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for linalg_op in _payload_ops(
            state, op.target, (linalg.TransposeOp, linalg.GenericOp)
        ):
            if _is_small_transpose(linalg_op):
                maps = linalg.get_indexing_maps(linalg_op)
                n_dims = ir.AffineMapAttr(maps[0]).value.n_dims
                linalg_op.operation.attributes[TILE_SIZES_ATTR_NAME] = (
                    ir.DenseI64ArrayAttr.get([0] * n_dims)
                )
        return DiagnosedSilenceableFailure.Success


def _is_small_transpose(op: ir.OpView) -> bool:
    """A static transpose of at most ``_MAX_VECTOR_ELEMENTS`` elements with no
    linalg producer or user, which tiling would fuse it with."""
    if not _is_transpose(op):
        return False
    if math.prod(ir.ShapedType(op.operands[1].type).shape) > _MAX_VECTOR_ELEMENTS:
        return False
    if _is_linalg(op.operands[0].owner):
        return False
    return not any(_is_linalg(use.owner) for use in op.results[0].uses)


def _is_transpose(op: ir.OpView) -> bool:
    """A static linalg op only moving its one input to a permuted layout."""
    if len(op.operands) != 2 or not _has_static_shape(op):
        return False
    maps = linalg.get_indexing_maps(op)
    if maps is None or len(maps) != 2:
        return False
    source, result = (ir.AffineMapAttr(affine_map).value for affine_map in maps)
    if source == result or not (source.is_permutation and result.is_permutation):
        return False
    body = list(op.regions[0].blocks[0].operations)
    return (
        len(body) == 1 and body[0].operands[0] == op.regions[0].blocks[0].arguments[0]
    )


def _is_linalg(owner: object) -> bool:
    if isinstance(owner, ir.Operation):
        owner = owner.opview
    return isinstance(owner, ir.OpView) and linalg.get_indexing_maps(owner) is not None


def _scalarize(
    transfer: vector.TransferReadOp | vector.TransferWriteOp,
    rewriter: transform.TransformRewriter,
) -> None:
    base_type = transfer.base.type
    if not isinstance(base_type, ir.MemRefType):
        return
    is_read = isinstance(transfer, vector.TransferReadOp)
    vector_type = ir.VectorType(
        transfer.result.type if is_read else transfer.valueToStore.type
    )
    is_bool = vector_type.element_type == ir.IntegerType.get_signless(1)
    minor_identity = ir.AffineMap.get_minor_identity(base_type.rank, vector_type.rank)
    if (vector_type.rank != 0 and not is_bool) or (
        transfer.permutation_map.value != minor_identity
    ):
        return
    in_bounds = [ir.BoolAttr(flag).value for flag in transfer.in_bounds]
    index = ir.IndexType.get()
    leading = base_type.rank - vector_type.rank
    with ir.InsertionPoint(transfer), transfer.location:
        extents = {
            dim: memref.DimOp(
                transfer.base, arith.ConstantOp(index, leading + dim)
            ).result
            for dim, flag in enumerate(in_bounds)
            if not flag
        }
        result = (
            vector.BroadcastOp(vector_type, transfer.padding).result
            if is_read
            else None
        )
        for position in itertools.product(*map(range, vector_type.shape)):
            indices = list(transfer.indices)
            checks = []
            for dim, offset in enumerate(position):
                if offset:
                    step = arith.ConstantOp(index, offset).result
                    indices[leading + dim] = arith.AddIOp(
                        indices[leading + dim], step
                    ).result
                if dim in extents:
                    checks.append(
                        arith.CmpIOp(
                            arith.CmpIPredicate.slt,
                            indices[leading + dim],
                            extents[dim],
                        ).result
                    )
            if transfer.mask is not None:
                checks.append(vector.extract(transfer.mask, [], list(position)))
            guard = None
            for check in checks:
                guard = check if guard is None else arith.AndIOp(guard, check).result
            if is_read:
                element = vector_type.element_type
                if guard is None:
                    scalar = memref.LoadOp(transfer.base, indices).result
                else:
                    branch = scf.IfOp(guard, [element], has_else=True)
                    with ir.InsertionPoint(branch.then_block):
                        scf.YieldOp([memref.LoadOp(transfer.base, indices).result])
                    with ir.InsertionPoint(branch.else_block):
                        scf.YieldOp([transfer.padding])
                    scalar = branch.results[0]
                result = vector.insert(scalar, result, [], list(position))
            else:
                scalar = vector.extract(transfer.valueToStore, [], list(position))
                if guard is None:
                    memref.StoreOp(scalar, transfer.base, indices)
                else:
                    branch = scf.IfOp(guard, [], has_else=False)
                    with ir.InsertionPoint(branch.then_block):
                        memref.StoreOp(scalar, transfer.base, indices)
                        scf.YieldOp([])
    if is_read:
        rewriter.replace_op(transfer, [result])
    else:
        rewriter.erase_op(transfer)


def _int_constant(value: ir.Value) -> int | None:
    if isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, arith.ConstantOp
    ):
        attr = value.owner.opview.value
        if isinstance(attr, ir.IntegerAttr):
            return attr.value
    return None


def _min(bounds: Sequence[int | None]) -> int | None:
    known = [bound for bound in bounds if bound is not None]
    return min(known) if known else None


def _size_bound(value: ir.Value, depth: int = 0) -> int | None:
    """A constant upper bound of the index ``value``, if one is evident."""
    if (constant := _int_constant(value)) is not None:
        return constant
    if depth > 16 or not isinstance(value, ir.OpResult):
        return None
    op = value.owner.opview
    if isinstance(op, affine.AffineMinOp):
        affine_map = ir.AffineMapAttr(op.attributes["map"]).value
        operands = list(op.operands)
        bounds = []
        for expr in affine_map.results:
            if isinstance(expr, ir.AffineConstantExpr):
                bounds.append(expr.value)
            elif isinstance(expr, ir.AffineDimExpr):
                operand = operands[expr.position]
                bounds.append(_size_bound(operand, depth + 1))
            elif isinstance(expr, ir.AffineSymbolExpr):
                operand = operands[affine_map.n_dims + expr.position]
                bounds.append(_size_bound(operand, depth + 1))
        return _min(bounds)
    if isinstance(op, arith.MinSIOp | arith.MinUIOp):
        return _min([_size_bound(operand, depth + 1) for operand in op.operands])
    if isinstance(op, arith.AddIOp):
        bounds = [_size_bound(operand, depth + 1) for operand in op.operands]
        return None if None in bounds else sum(bounds)
    if isinstance(op, tensor.DimOp) and (dim := _int_constant(op.index)) is not None:
        return _dim_bound(op.source, dim, depth + 1)
    return None


def _dynamic_size(
    sizes: Sequence[ir.Value], shaped: ir.ShapedType, dim: int
) -> ir.Value:
    """The size operand of dynamic ``dim`` of an op listing only dynamic sizes."""
    return sizes[sum(shaped.is_dynamic_dim(i) for i in range(dim))]


def _pad_amount(pad: tensor.PadOp, high: bool, dim: int) -> int | ir.Value:
    """The low or high padding of ``dim``, as a constant or a value."""
    static = list(pad.static_high if high else pad.static_low)
    if static[dim] != ir.ShapedType.get_dynamic_size():
        return static[dim]
    dynamic = list(pad.high if high else pad.low)
    return dynamic[
        sum(size == ir.ShapedType.get_dynamic_size() for size in static[:dim])
    ]


def _pad_amount_bound(
    pad: tensor.PadOp, high: bool, dim: int, depth: int
) -> int | None:
    amount = _pad_amount(pad, high, dim)
    return amount if isinstance(amount, int) else _size_bound(amount, depth)


def _dim_bound(value: ir.Value, dim: int, depth: int = 0) -> int | None:
    """A constant upper bound of ``dim`` of the tensor ``value``, if one is evident."""
    shaped = ir.ShapedType(value.type)
    if not shaped.is_dynamic_dim(dim):
        return shaped.get_dim_size(dim)
    if depth > 16:
        return None
    if isinstance(value, ir.BlockArgument):
        owner = value.owner.owner.operation.opview
        if isinstance(owner, scf.ForOp):
            return _dim_bound(owner.initArgs[value.arg_number - 1], dim, depth + 1)
        if isinstance(owner, scf.ForallOp):
            ivs = len(owner.induction_variables)
            return _dim_bound(owner.outputs[value.arg_number - ivs], dim, depth + 1)
        return None
    op = value.owner.opview
    if isinstance(op, tensor.ExtractSliceOp | tensor.EmptyOp | tensor.GenerateOp):
        sizes = {
            tensor.ExtractSliceOp: lambda: op.sizes,
            tensor.EmptyOp: lambda: op.dynamicSizes,
            tensor.GenerateOp: lambda: op.dynamicExtents,
        }[type(op)]()
        # Dims an extract_slice drops are unit dims, so dynamic sizes stay in order.
        return _size_bound(_dynamic_size(list(sizes), shaped, dim), depth + 1)
    if isinstance(op, tensor.CastOp):
        return _dim_bound(op.source, dim, depth + 1)
    if isinstance(op, tensor.PadOp):
        bounds = [
            _dim_bound(op.source, dim, depth + 1),
            *(_pad_amount_bound(op, high, dim, depth + 1) for high in (False, True)),
        ]
        return None if None in bounds else sum(bounds)
    if isinstance(op, tensor.InsertSliceOp):
        return _dim_bound(op.dest, dim, depth + 1)
    if isinstance(op, vector.TransferWriteOp):
        return _dim_bound(op.base, dim, depth + 1)
    if isinstance(op, scf.ForOp):
        return _dim_bound(op.initArgs[value.result_number], dim, depth + 1)
    if isinstance(op, scf.ForallOp):
        return _dim_bound(op.outputs[value.result_number], dim, depth + 1)
    if isinstance(op, scf.IfOp):
        if any(not region.blocks for region in op.regions):
            return None
        yields = [
            list(region.blocks[0].operations)[-1].operands[value.result_number]
            for region in op.regions
        ]
        bounds = [_dim_bound(yielded, dim, depth + 1) for yielded in yields]
        return None if None in bounds else max(bounds)
    if linalg.get_indexing_maps(op) is not None:
        outputs = list(op.operands)[len(op.operands) - len(op.results) :]
        return _dim_bound(outputs[value.result_number], dim, depth + 1)
    return None


def _loop_bounds(op: ir.OpView) -> list[int | None] | None:
    """Constant upper bounds of a linalg op's loop ranges (``None`` where not
    evident), or ``None`` for an op without indexing maps."""
    maps = linalg.get_indexing_maps(op)
    operands = list(op.operands)
    if maps is None or len(maps) != len(operands):
        return None
    maps = [ir.AffineMapAttr(affine_map).value for affine_map in maps]
    bounds: list[int | None] = [None] * maps[0].n_dims
    for operand, affine_map in zip(operands, maps, strict=True):
        if not isinstance(operand.type, ir.ShapedType):
            continue
        for position, expr in enumerate(affine_map.results):
            if isinstance(expr, ir.AffineDimExpr):
                loop_dim = expr.position
                bounds[loop_dim] = _min(
                    [bounds[loop_dim], _dim_bound(operand, position)]
                )
    return bounds


def _tile_sizes(op: ir.OpView) -> list[int] | None:
    """Tile sizes making a linalg op's loop ranges bounded and its vectors at most
    ``_MAX_VECTOR_ELEMENTS``, those of an op of runtime shape multiples of
    ``_TILE``, and those of a transpose 1 on all but its source's two inner dims
    (0: untiled)."""
    bounds = _loop_bounds(op)
    if bounds is None:
        return None
    too_large = None not in bounds and math.prod(bounds) > _MAX_VECTOR_ELEMENTS
    static = _has_static_shape(op)
    sizes = [
        _TILE
        if bound is None
        or (bound > _TILE and (too_large or (not static and bound % _TILE)))
        else 0
        for bound in bounds
    ]
    # LLVM takes many seconds on n-D vector transposes; 2-D ones are shuffles.
    if _is_transpose(op):
        source_map = ir.AffineMapAttr(linalg.get_indexing_maps(op)[0]).value
        for expr in list(source_map.results)[:-2]:
            dim = ir.AffineDimExpr(expr).position
            if bounds[dim] > 1:
                sizes[dim] = 1
    return sizes


def _has_static_shape(op: ir.OpView) -> bool:
    return all(
        ir.ShapedType(value.type).has_static_shape
        for value in [*op.operands, *op.results]
        if isinstance(value.type, ir.ShapedType)
    )


# Loop counts of the ops vectorize_linalg vectorizes with masks.
_MAX_LOOPS = 8
# Tile size of runtime extents without an evident bound, of odd runtime bounds, and
# of all extents of ops whose vectors would exceed _MAX_VECTOR_ELEMENTS: LLVM takes
# many seconds on large vectors and on odd-width masked ones.
_TILE = 32
_MAX_VECTOR_ELEMENTS = 4096


@_transform_op(modifies_payload=False)
class PartitionLinalgOp(HelionTransformDialect.Operation, name="partition_linalg"):
    """Group the target's linalg ops with loops: those needing tiling first
    (``_tile_sizes``); the rest of static shape; and those of runtime shape with
    evident loop bounds, by loop count (1 to ``_MAX_LOOPS``). Upstream vectorizes
    a loop-free op reading with ``tensor.extract`` into invalid IR."""

    target: ext.Operand[transform.AnyOpType]
    groups: Sequence[ext.Result[transform.AnyOpType]]

    @staticmethod
    def run(
        op: "PartitionLinalgOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        groups: list[list[ir.Operation]] = [[] for _ in op.groups]

        def collect(visited: ir.Operation) -> ir.WalkResult:
            bounds = _loop_bounds(visited.opview)
            if not bounds:
                return ir.WalkResult.ADVANCE
            if any(_tile_sizes(visited.opview)):
                groups[-1].append(visited)
            elif _has_static_shape(visited.opview):
                groups[0].append(visited)
            elif 0 < len(bounds) < len(groups) - 1:
                groups[len(bounds)].append(visited)
            return ir.WalkResult.ADVANCE

        for target in state.get_payload_ops(op.target):
            target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
        for handle, ops in zip(op.groups, groups, strict=True):
            results.set_ops(handle, ops)
        return DiagnosedSilenceableFailure.Success


@_transform_op(modifies_payload=False)
class LoopBoundsOp(HelionTransformDialect.Operation, name="loop_bounds"):
    """The constant upper bounds of one linalg op's loop ranges, one param each."""

    target: ext.Operand[transform.AnyOpType]
    bounds: Sequence[ext.Result[transform.AnyParamType]]

    @staticmethod
    def run(
        op: "LoopBoundsOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        bounds = _loop_bounds(targets[0].opview) if len(targets) == 1 else None
        if bounds is None or None in bounds or len(bounds) != len(op.bounds):
            return DiagnosedSilenceableFailure.SilenceableFailure
        i64 = ir.IntegerType.get_signless(64)
        for handle, bound in zip(op.bounds, bounds, strict=True):
            results.set_params(handle, [ir.IntegerAttr.get(i64, bound)])
        return DiagnosedSilenceableFailure.Success


@_transform_op(modifies_payload=False)
class TileSizesOp(HelionTransformDialect.Operation, name="tile_sizes"):
    """The ``_tile_sizes`` of one linalg op, as one param."""

    target: ext.Operand[transform.AnyOpType]
    sizes: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @staticmethod
    def run(
        op: "TileSizesOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        targets = state.get_payload_ops(op.target)
        sizes = _tile_sizes(targets[0].opview) if len(targets) == 1 else None
        if sizes is None:
            return DiagnosedSilenceableFailure.SilenceableFailure
        i64 = ir.IntegerType.get_signless(64)
        results.set_params(op.sizes, [ir.IntegerAttr.get(i64, s) for s in sizes])
        return DiagnosedSilenceableFailure.Success


def _mask_sizes(mask: ir.Value) -> list[ir.Value | int] | None:
    """The per-dim sizes of a ``vector.create_mask``/``constant_mask`` value."""
    if not isinstance(mask, ir.OpResult):
        return None
    op = mask.owner.opview
    if isinstance(op, vector.CreateMaskOp):
        return list(op.operands)
    if isinstance(op, vector.ConstantMaskOp):
        return list(op.mask_dim_sizes)
    return None


def _unmask_contraction(
    masked: vector.MaskOp, rewriter: transform.TransformRewriter
) -> None:
    body = list(masked.maskRegion.blocks[0].operations)
    contract = body[0].opview if len(body) == 2 else None
    sizes = _mask_sizes(masked.mask)
    if (
        not isinstance(contract, vector.ContractionOp)
        or masked.passthru is not None
        or sizes is None
        or str(contract.kind) != "#vector.kind<add>"
        or not all(
            isinstance(ir.VectorType(value.type).element_type, ir.FloatType)
            for value in (contract.lhs, contract.rhs)
        )
    ):
        return
    maps = [ir.AffineMapAttr(affine_map).value for affine_map in contract.indexing_maps]
    if len(sizes) != maps[0].n_dims:
        return
    index = ir.IndexType.get()
    with ir.InsertionPoint(masked), masked.location:
        sizes = [
            size if isinstance(size, ir.Value) else arith.ConstantOp(index, size).result
            for size in sizes
        ]
        operands = []
        for operand, affine_map in zip(
            (contract.lhs, contract.rhs), maps, strict=False
        ):
            dims = [expr.position for expr in affine_map.results]
            vector_type = ir.VectorType(operand.type)
            mask = vector.CreateMaskOp(
                ir.VectorType.get(vector_type.shape, ir.IntegerType.get_signless(1)),
                [sizes[dim] for dim in dims],
            ).result
            zero = arith.ConstantOp(
                vector_type,
                ir.DenseElementsAttr.get_splat(
                    vector_type, ir.FloatAttr.get(vector_type.element_type, 0.0)
                ),
            ).result
            operands.append(arith.SelectOp(mask, operand, zero).result)
        result = vector.ContractionOp(
            contract.result.type,
            *operands,
            contract.acc,
            contract.indexing_maps,
            contract.iterator_types,
            kind=contract.kind,
            fastmath=contract.fastmath,
        ).result
    rewriter.replace_op(masked, [result])


@_transform_op(modifies_payload=True)
class UnmaskContractionsOp(
    HelionTransformDialect.Operation, name="unmask_contractions"
):
    """Rewrite every masked floating-point add-contraction in the target as an
    unmasked one of operands zeroed where masked off."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "UnmaskContractionsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for masked in _payload_ops(state, op.target, vector.MaskOp):
            _unmask_contraction(masked, rewriter)
        return DiagnosedSilenceableFailure.Success


def _extend_contraction(
    contract: vector.ContractionOp, rewriter: transform.TransformRewriter
) -> None:
    acc_type = contract.acc.type
    element = (
        ir.VectorType(acc_type).element_type
        if isinstance(acc_type, ir.VectorType)
        else acc_type
    )
    operand_types = [
        ir.VectorType(value.type) for value in (contract.lhs, contract.rhs)
    ]
    if not isinstance(element, ir.FloatType) or all(
        operand.element_type == element for operand in operand_types
    ):
        return
    if not all(
        isinstance(operand.element_type, ir.FloatType) for operand in operand_types
    ):
        return
    with ir.InsertionPoint(contract), contract.location:
        operands = [
            value
            if operand.element_type == element
            else arith.ExtFOp(ir.VectorType.get(operand.shape, element), value).result
            for value, operand in zip(
                (contract.lhs, contract.rhs), operand_types, strict=True
            )
        ]
        result = vector.ContractionOp(
            contract.result.type,
            *operands,
            contract.acc,
            contract.indexing_maps,
            contract.iterator_types,
            kind=contract.kind,
            fastmath=contract.fastmath,
        ).result
    rewriter.replace_op(contract, [result])


@_transform_op(modifies_payload=True)
class LegalizeForLLVMOp(HelionTransformDialect.Operation, name="legalize_for_llvm"):
    """Rewrite every 0-d or i1 minor-identity memref transfer in the target as
    per-element scalar loads or stores, and every floating-point contraction with
    operands narrower than its accumulator as one of extended operands."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "LegalizeForLLVMOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        transfer_types = (vector.TransferReadOp, vector.TransferWriteOp)
        for transfer in _payload_ops(state, op.target, transfer_types):
            _scalarize(transfer, rewriter)
        for contract in _payload_ops(state, op.target, vector.ContractionOp):
            _extend_contraction(contract, rewriter)
        return DiagnosedSilenceableFailure.Success


@contextmanager
def _suppressing(op: ir.Value) -> Iterator[ir.Value]:
    """A sequence on ``op`` whose silenceable failures are ignored."""
    sequence = transform.SequenceOp(transform.FailurePropagationMode.Suppress, [], op)
    with ir.InsertionPoint(sequence.body):
        yield sequence.bodyTarget
        transform.yield_()


def vectorize_linalg() -> ir.Module:
    """Schedule: lighthouse's ``vectorization.py[gen=vectorize_linalg]``, also
    vectorizing ops of runtime shape, with masks. Runtime extents without an
    evident bound and ops of too large vectors are first tiled. Ops that cannot
    be tiled or vectorized (e.g. argmax, gathers) are left to the loop lowering."""
    HelionTransformDialect.load()
    groups = [transform.AnyOpType.get()] * (_MAX_LOOPS + 2)
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        to_tile = PartitionLinalgOp(target=funcs, groups=groups).groups[-1]
        with lh_transform.foreach(to_tile) as op:
            sizes = TileSizesOp(target=op).sizes
            with _suppressing(op) as target:
                structured.TileUsingForOp(target, sizes=sizes)
            transform.yield_()
        static, *dynamic, _ = PartitionLinalgOp(target=funcs, groups=groups).groups
        with lh_transform.foreach(static) as op:
            with _suppressing(op) as target:
                structured.structured_vectorize(
                    target, [], create_named_contraction=True
                )
            transform.yield_()
        for loops, group in enumerate(dynamic, 1):
            with lh_transform.foreach(group) as op:
                with _suppressing(op) as target:
                    bounds = LoopBoundsOp(
                        target=target,
                        bounds=[transform.AnyParamType.get()] * loops,
                    ).bounds
                    structured.structured_vectorize(
                        target,
                        list(bounds),
                        static_vector_sizes=[ir.ShapedType.get_dynamic_size()] * loops,
                        scalable_sizes=[False] * loops,
                        create_named_contraction=True,
                    )
                transform.yield_()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # Masked transfers as transfers with a mask operand: later patterns
            # would otherwise build ops inside vector.mask regions.
            transform_vector.apply_patterns_vector_lower_masked_transfers()
            transform_vector.apply_patterns_vector_reduction_to_contract()
            transform_vector.apply_patterns_vector_transfer_permutation_patterns()
            transform_vector.apply_patterns_vector_fold_arith_extension()
        # x86 contraction patterns would rewrite a masked one inside its mask.
        UnmaskContractionsOp(target=funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def vectorize_pads() -> ir.Module:
    """Schedule: vectorize the pads of every function."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        VectorizePadsOp(target=funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def materialize_copies() -> ir.Module:
    """Schedule: large slice moves of every function as ``linalg.copy`` ops."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        MaterializeCopiesOp(target=funcs)
        transform.yield_()
    return schedule


def split_transfers() -> ir.Module:
    """Schedule: split every function's out-of-bounds memref transfers on an
    in-bounds check."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        SplitTransfersOp(target=funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def pin_transposes() -> ir.Module:
    """Schedule: keep every function's small static transposes untiled."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        PinTransposesOp(target=funcs)
        transform.yield_()
    return schedule


def legalize_for_llvm() -> ir.Module:
    """Schedule: legalize every function's vector ops for the LLVM lowering."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        LegalizeForLLVMOp(target=funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
