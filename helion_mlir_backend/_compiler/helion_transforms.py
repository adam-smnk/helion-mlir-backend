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
import operator

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
from mlir.dialects.transform import tensor as transform_tensor
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


def _pad_value(pad: tensor.PadOp | tensor.GenerateOp) -> ir.Value | None:
    """The pad's (or generate's) constant value, usable before it, if it has one."""
    body = pad.regions[0].blocks[0]
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
    target of a static slice of more than ``_MAX_VECTOR_ELEMENTS`` elements, and
    a row-by-row copy for a runtime-shaped slice."""

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
    if not source_type.has_static_shape:
        _materialize_row_copy(insert, source)
        return
    if math.prod(source_type.shape) <= _MAX_VECTOR_ELEMENTS or [
        size for size in sizes if size != 1
    ] != [dim for dim in source_type.shape if dim != 1]:
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


def _materialize_row_copy(insert: tensor.InsertSliceOp, source: ir.Value) -> None:
    """Copy a runtime-shaped ``source`` into the destination slice of
    ``insert`` row by row (see ``_copy_rows``): a bufferized ``memref.copy``
    of a strided slice is an element-wise runtime library call."""
    source_type = ir.RankedTensorType(source.type)
    if insert.source != source or source_type.rank != len(insert.static_sizes):
        return
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
        copied = _copy_rows(source, destination, _sizes(source))
    insert.operation.operands[0] = copied


def _sizes(value: ir.Value) -> list[ir.Value | int]:
    value_type = ir.RankedTensorType(value.type)
    index_type = ir.IndexType.get()
    return [
        tensor.DimOp(value, arith.ConstantOp(index_type, dim).result).result
        if value_type.is_dynamic_dim(dim)
        else value_type.shape[dim]
        for dim in range(value_type.rank)
    ]


def _copy_rows(
    source: ir.Value,
    dest: ir.Value,
    sizes: list[ir.Value | int],
    padding: ir.Value | None = None,
) -> ir.Value:
    """Copy ``source`` of ``sizes`` into the leading corner of ``dest`` one
    innermost row at a time: 1-D copies vectorize to plain masked loads and
    stores, n-D ones to transfers lowered through memory. With ``padding``, every
    row of the static ``dest`` is written: a row of ``source`` padded past its
    end, other rows filled."""
    index_type = ir.IndexType.get()
    dynamic = ir.ShapedType.get_dynamic_size()
    rank = len(sizes)
    cols = sizes[-1]
    static_cols = cols if isinstance(cols, int) else dynamic
    dynamic_cols = [] if isinstance(cols, int) else [cols]
    element_type = ir.RankedTensorType(source.type).element_type
    row_type = ir.RankedTensorType.get([static_cols], element_type)
    dest_shape = ir.RankedTensorType(dest.type).shape
    width = dest_shape[-1]
    zero = arith.ConstantOp(index_type, 0).result
    one = arith.ConstantOp(index_type, 1).result

    def constant(size: ir.Value | int) -> ir.Value:
        return (
            size
            if not isinstance(size, int)
            else arith.ConstantOp(index_type, size).result
        )

    def write_row(buffer: ir.Value, rows: list[ir.Value]) -> ir.Value:
        args = (
            rows,
            dynamic_cols,
            [],
            [dynamic] * len(rows) + [0],
            [1] * len(rows) + [static_cols],
            [1] * rank,
        )
        source_row = tensor.extract_slice(row_type, source, *args)
        if padding is None or static_cols == width:
            dest_row = tensor.extract_slice(row_type, buffer, *args)
            copied = linalg.copy(source_row, outs=[dest_row])
            return tensor.insert_slice(copied, buffer, *args)
        high = arith.SubIOp(
            arith.ConstantOp(index_type, width).result,
            tensor.DimOp(source_row, zero).result,
        ).result
        padded = tensor.PadOp(
            ir.RankedTensorType.get([width], element_type),
            source_row,
            [],
            [high],
            [0],
            [dynamic],
        )
        body = padded.regions[0].blocks.append(index_type)
        with ir.InsertionPoint(body):
            tensor.YieldOp(padding)
        return tensor.insert_slice(padded.result, buffer, *full_row(rows))

    def full_row(rows: list[ir.Value]) -> tuple:
        return (
            rows,
            [],
            [],
            [dynamic] * len(rows) + [0],
            [1] * len(rows) + [width],
            [1] * rank,
        )

    def fill_row(buffer: ir.Value, rows: list[ir.Value]) -> ir.Value:
        dest_row = tensor.extract_slice(
            ir.RankedTensorType.get([width], element_type), buffer, *full_row(rows)
        )
        filled = linalg.fill(padding, outs=[dest_row])
        return tensor.insert_slice(filled, buffer, *full_row(rows))

    def copy(buffer: ir.Value, rows: list[ir.Value]) -> ir.Value:
        if len(rows) == rank - 1:
            inside = None
            if padding is not None:
                for row, size, extent in zip(rows, sizes, dest_shape, strict=False):
                    if size == extent:
                        continue
                    row_inside = arith.CmpIOp(
                        arith.CmpIPredicate.ult, row, constant(size)
                    ).result
                    inside = (
                        row_inside
                        if inside is None
                        else arith.AndIOp(inside, row_inside).result
                    )
            if inside is None:
                return write_row(buffer, rows)
            branch = scf.IfOp(inside, [buffer.type], has_else=True)
            with ir.InsertionPoint(branch.then_block):
                scf.YieldOp([write_row(buffer, rows)])
            with ir.InsertionPoint(branch.else_block):
                scf.YieldOp([fill_row(buffer, rows)])
            return branch.results[0]
        bound = constant(sizes[len(rows)] if padding is None else dest_shape[len(rows)])
        loop = scf.ForOp(zero, bound, one, [buffer])
        with ir.InsertionPoint(loop.body):
            scf.YieldOp(
                [copy(loop.inner_iter_args[0], [*rows, loop.induction_variable])]
            )
        return loop.results[0]

    return copy(dest, [])


def _feeds_contraction(value: ir.Value, packs: bool = True) -> bool:
    """Whether every use of ``value`` reads it, possibly reshaped or, with
    ``packs``, transposed (e.g. packed), as a contraction input."""
    uses = list(value.uses)
    for use in uses:
        owner = _opview(use.owner)
        if isinstance(owner, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)) or (
            packs
            and _is_linalg(owner)
            and _is_transpose(owner)
            and use.operand_number == 0
        ):
            if not _feeds_contraction(owner.results[0], packs):
                return False
        elif not _is_contraction_input(use):
            return False
    return bool(uses)


def _materialize_pad(pad: tensor.PadOp, packed_only: bool) -> None:
    """A static zero-high-padded contraction operand as a filled tensor its
    source is copied into row by row: kernels like AMX load operand tiles from
    memory, not from vectors of masked reads. With ``packed_only``, only one read
    through an operand pack: register tiling fuses one read directly."""
    result_type = ir.RankedTensorType(pad.result.type)
    if (
        list(pad.low)
        or any(pad.static_low)
        or not result_type.has_static_shape
        or not _feeds_contraction(pad.result)
        or (packed_only and _feeds_contraction(pad.result, packs=False))
    ):
        return
    with ir.InsertionPoint(pad), pad.location:
        padding = _pad_value(pad)
        if padding is None:
            return
        shape = list(result_type.shape)
        sizes = _sizes(pad.source)
        # Rows are the trailing dims past the last padded one, collapsed: a
        # reshape may have left full dims inner to it (e.g. VNNI pairs).
        last = len(shape) - 1
        while last > 0 and sizes[last] == shape[last]:
            last -= 1
        inner = list(range(last + 1, len(shape)))
        groups = [[dim] for dim in range(last + 1)] + [inner]
        source = pad.source
        if len(inner) > 1:
            source_shape = ir.RankedTensorType(source.type).shape
            source = tensor.CollapseShapeOp(
                ir.RankedTensorType.get(
                    [*source_shape[: last + 1], math.prod(shape[last + 1 :])],
                    result_type.element_type,
                ),
                source,
                groups,
            ).result
        rows_shape = (
            [*shape[: last + 1], math.prod(shape[last + 1 :])]
            if len(inner) > 1
            else shape
        )
        empty = tensor.EmptyOp(rows_shape, result_type.element_type).result
        copied = _copy_rows(source, empty, _sizes(source), padding)
        if len(inner) > 1:
            copied = tensor.ExpandShapeOp(
                result_type, copied, groups, [], static_output_shape=shape
            ).result
    pad.result.replace_all_uses_with(copied)
    pad.operation.erase()


def _materialize_generate(generate: tensor.GenerateOp) -> None:
    """A static constant contraction operand (e.g. tiling's all-padding tile)
    filled one row, its trailing dims collapsed, at a time: register tiling
    unrolls a whole fill into one op per vector."""
    result_type = ir.RankedTensorType(generate.result.type)
    rank = result_type.rank
    if (
        not result_type.has_static_shape
        or rank < 2
        or not _feeds_contraction(generate.result)
    ):
        return
    with ir.InsertionPoint(generate), generate.location:
        value = _pad_value(generate)
        if value is None:
            return
        shape = list(result_type.shape)
        rows, width = shape[0], math.prod(shape[1:])
        element_type = result_type.element_type
        index_type = ir.IndexType.get()
        empty = tensor.EmptyOp([rows, width], element_type).result
        loop = scf.ForOp(
            arith.ConstantOp(index_type, 0).result,
            arith.ConstantOp(index_type, rows).result,
            arith.ConstantOp(index_type, 1).result,
            [empty],
        )
        with ir.InsertionPoint(loop.body):
            buffer = loop.inner_iter_args[0]
            row = (
                [loop.induction_variable],
                [],
                [],
                [ir.ShapedType.get_dynamic_size(), 0],
                [1, width],
                [1, 1],
            )
            row_type = ir.RankedTensorType.get([width], element_type)
            filled = linalg.fill(
                value, outs=[tensor.extract_slice(row_type, buffer, *row)]
            )
            scf.YieldOp([tensor.insert_slice(filled, buffer, *row)])
        filled = loop.results[0]
        if rank > 2:
            filled = tensor.ExpandShapeOp(
                result_type,
                filled,
                [[0], list(range(1, rank))],
                [],
                static_output_shape=shape,
            ).result
    generate.result.replace_all_uses_with(filled)
    generate.operation.erase()


@_transform_op(modifies_payload=True)
class MaterializeOperandPadsOp(
    HelionTransformDialect.Operation, name="materialize_operand_pads"
):
    """Materialize every zero-high-padded contraction operand in the target
    (see ``_materialize_pad``) and, unless ``packed_only``, every constant one
    (see ``_materialize_generate``)."""

    target: ext.Operand[transform.AnyOpType]
    packed_only: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
    )

    @staticmethod
    def run(
        op: "MaterializeOperandPadsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        packed_only = bool(ir.IntegerAttr(op.packed_only).value)
        for pad in _payload_ops(state, op.target, tensor.PadOp):
            _materialize_pad(pad, packed_only)
        if not packed_only:
            for generate in _payload_ops(state, op.target, tensor.GenerateOp):
                _materialize_generate(generate)
        return DiagnosedSilenceableFailure.Success


# Ops of these dialects have no memory effects.
_PURE_DIALECTS = ("tensor.", "arith.", "affine.")

# An operand read through reshapes: (op, operand index of the value read) from
# the first reshape to the consumer.
_Chain = list[tuple[ir.OpView, int]]


def _runtime_padding(pad: tensor.PadOp) -> list[ir.Value] | None:
    """The runtime padding amounts of ``pad`` if all others are zero, else ``None``."""
    dynamic = ir.ShapedType.get_dynamic_size()
    if any(
        amount not in (0, dynamic) for amount in [*pad.static_low, *pad.static_high]
    ):
        return None
    return [*pad.low, *pad.high] or None


def _operand_chain(consumer: ir.OpView, index: int) -> tuple[_Chain, ir.Value]:
    """The reshapes in ``consumer``'s block, each only read by the next, that
    operand ``index`` of ``consumer`` is read through; the value reshaped."""
    chain = [(consumer, index)]
    value = consumer.operands[index]
    while isinstance(value, ir.OpResult):
        op = value.owner.opview
        if (
            not isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp))
            or op.operation.block != consumer.operation.block
            or len(list(value.uses)) != 1
        ):
            break
        chain.insert(0, (op, 0))
        value = op.operands[0]
    return chain, value


def _only_read_by(value: ir.Value, chain: _Chain) -> ir.OpView | None:
    """The op defining ``value`` if it is only read by the first op of
    ``chain``, in its block."""
    if not isinstance(value, ir.OpResult) or len(list(value.uses)) != 1:
        return None
    op = value.owner.opview
    return op if op.operation.block == chain[0][0].operation.block else None


def _versionable_pad(value: ir.Value, chain: _Chain) -> tensor.PadOp | None:
    """The pad of runtime padding and static shape defining ``value``, only
    read by ``chain``, else ``None``."""
    pad = _only_read_by(value, chain)
    if (
        not isinstance(pad, tensor.PadOp)
        or not ir.RankedTensorType(pad.result.type).has_static_shape
        or _runtime_padding(pad) is None
    ):
        return None
    return pad


def _sinkable_branch(value: ir.Value, chain: _Chain) -> scf.IfOp | None:
    """The side-effect-free ``scf.if`` defining ``value``, only read by
    ``chain``, with a branch yielding a pad (e.g. tiling's guard of a pad's
    empty slices), else ``None``."""
    branch = _only_read_by(value, chain)
    if (
        not isinstance(branch, scf.IfOp)
        or len(branch.results) != 1
        or len(branch.regions[1].blocks) != 1
    ):
        return None
    padded = False
    for region in branch.regions:
        ops = list(region.blocks[0].operations)
        if not all(op.name.startswith(_PURE_DIALECTS) for op in ops[:-1]):
            return None
        yielded = ops[-1].operands[0]
        padded |= isinstance(yielded, ir.OpResult) and isinstance(
            yielded.owner.opview, tensor.PadOp
        )
    return branch if padded else None


def _clone_chain(chain: _Chain, value: ir.Value) -> list[ir.OpView]:
    """Copies of ``chain`` at the insertion point reading ``value``."""
    copies = []
    for op, index in chain:
        copy = op.operation.clone()
        copy.operation.operands[index] = value
        value = copy.results[0]
        copies.append(copy)
    return copies


def _replace(chain: _Chain, branch: scf.IfOp) -> None:
    consumer = chain[-1][0]
    for old, new in zip(consumer.results, branch.results, strict=True):
        old.replace_all_uses_with(new)
    for op, _ in reversed(chain):
        op.operation.erase()


def _sink_into_branch(chain: _Chain, branch: scf.IfOp) -> list[ir.OpView]:
    """Move ``branch``, read by ``chain``, to its consumer and ``chain`` into
    each of its branches; the consumer's copies."""
    consumer = chain[-1][0]
    with ir.InsertionPoint(consumer), consumer.location:
        sunk = scf.IfOp(
            branch.condition, [r.type for r in consumer.results], has_else=True
        )
    copies = []
    for region, block in zip(
        branch.regions, (sunk.then_block, sunk.else_block), strict=True
    ):
        ops = list(region.blocks[0].operations)
        with ir.InsertionPoint(block), consumer.location:
            cloned = _clone_chain(chain, ops[-1].operands[0])
            for op in ops[:-1]:
                op.move_before(cloned[0])
            scf.YieldOp(list(cloned[-1].results))
        copies.append(cloned[-1])
    _replace(chain, sunk)
    branch.operation.erase()
    return copies


def _version_on_pad(chain: _Chain, pad: tensor.PadOp) -> list[ir.OpView]:
    """Branch ``chain``'s consumer on ``pad``, read by ``chain``, padding
    nothing at runtime: then it reads the pad's source in place; the copies."""
    consumer = chain[-1][0]
    with ir.InsertionPoint(consumer), consumer.location:
        zero = arith.ConstantOp(ir.IndexType.get(), 0).result
        unpadded = None
        for amount in _runtime_padding(pad):
            is_zero = arith.CmpIOp(arith.CmpIPredicate.eq, amount, zero).result
            unpadded = (
                is_zero if unpadded is None else arith.AndIOp(unpadded, is_zero).result
            )
        branch = scf.IfOp(unpadded, [r.type for r in consumer.results], has_else=True)
    with ir.InsertionPoint(branch.then_block), consumer.location:
        # The source has the pad's shape when it pads nothing.
        source = tensor.CastOp(pad.result.type, pad.source).result
        in_place = _clone_chain(chain, source)
        scf.YieldOp(list(in_place[-1].results))
    with ir.InsertionPoint(branch.else_block), consumer.location:
        padded = _clone_chain(chain, pad.result)
        pad.operation.move_before(padded[0])
        scf.YieldOp(list(padded[-1].results))
    _replace(chain, branch)
    return [in_place[-1], padded[-1]]


def _version_operands(consumer: ir.OpView, done: frozenset[int] = frozenset()) -> None:
    """Version ``consumer`` on each runtime-padded input not in ``done`` (see
    ``_version_on_pad``), first sunk into the branches defining it, so each
    reads its pad directly."""
    for index in range(len(consumer.operands) - len(consumer.results)):
        if index in done:
            continue
        chain, value = _operand_chain(consumer, index)
        branch = _sinkable_branch(value, chain)
        if branch is not None:
            for copy in _sink_into_branch(chain, branch):
                _version_operands(copy, done)
            return
        pad = _versionable_pad(value, chain)
        if pad is not None:
            for copy in _version_on_pad(chain, pad):
                _version_operands(copy, done | {index})
            return


@_transform_op(modifies_payload=True)
class VersionPaddedOperandsOp(
    HelionTransformDialect.Operation, name="version_padded_operands"
):
    """Branch every contraction in the target on each of its runtime-padded
    inputs padding nothing: tiles of a partial tile but the edge ones read the
    source in place, and only the edge tiles are padded."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "VersionPaddedOperandsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        contractions = [
            found
            for found in _payload_ops(state, op.target, ir.OpView)
            if _is_linalg(found)
            and linalg.isa_contraction_op(found)
            and len(found.results) == 1
        ]
        for contraction in contractions:
            _version_operands(contraction)
        return DiagnosedSilenceableFailure.Success


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
    """Annotate every ``_is_small_transpose`` or ``_is_operand_pack`` op in the
    target with zero tile sizes: lighthouse's tiling keeps annotated sizes, so it
    stays one op, a fusion boundary."""

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
            if _is_small_transpose(linalg_op) or _is_operand_pack(linalg_op):
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


def _is_operand_pack(op: ir.OpView) -> bool:
    """A static transpose only read as a contraction operand: a pack into the
    layout a contraction kernel reads (e.g. VNNI). Fused into the contraction's
    tiles, its values would not be in memory, which AMX tile loads need."""
    if not _is_transpose(op):
        return False
    uses = list(op.results[0].uses)
    return bool(uses) and all(_is_contraction_input(use) for use in uses)


def _is_contraction_input(use: ir.OpOperand) -> bool:
    """Whether ``use`` reads its value, possibly through slices, as an input of
    a contraction."""
    owner = _opview(use.owner)
    if isinstance(owner, tensor.ExtractSliceOp):
        uses = list(owner.result.uses)
        return bool(uses) and all(_is_contraction_input(u) for u in uses)
    return (
        _is_linalg(owner)
        and linalg.isa_contraction_op(owner)
        and use.operand_number < len(owner.operands) - 1
    )


def _opview(owner: object) -> ir.OpView:
    return owner.opview if isinstance(owner, ir.Operation) else owner


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
    ``_TILE``, and those of a transpose 1 on all but the inner dims of its source
    and result (0: untiled)."""
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
    # Keeping the inner dims of both sides makes each tile contiguous reads
    # interleaved into contiguous writes (e.g. a VNNI pack: 2 rows into pairs).
    if _is_transpose(op):
        source_map, result_map = (
            ir.AffineMapAttr(affine_map).value
            for affine_map in linalg.get_indexing_maps(op)
        )
        source = [ir.AffineDimExpr(e).position for e in source_map.results]
        result_inner = ir.AffineDimExpr(result_map.results[-1]).position
        kept = {source[-1], result_inner}
        if len(kept) == 1 and len(source) > 1:
            kept.add(source[-2])
        for dim in source:
            if dim not in kept and bounds[dim] > 1:
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


@_transform_op(modifies_payload=True)
class HoistAllocasOp(HelionTransformDialect.Operation, name="hoist_allocas"):
    """Move every static ``memref.alloca`` in the target out of the loops of its
    ``alloca_scope``, ``omp.parallel`` region or function: in a loop, its
    lowering grows the stack every iteration until the scope ends. It goes to
    the entry of the block holding the outermost such loop, so buffers of
    exclusive ``scf.if`` branches above the loops do not add up."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "HoistAllocasOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        scopes = ("memref.alloca_scope", "omp.parallel", "func.func")
        loops = ("scf.for", "scf.while", "scf.parallel", "scf.forall")
        for alloca in _payload_ops(state, op.target, memref.AllocaOp):
            if list(alloca.operands):
                continue
            outermost = None
            parent = alloca.operation.parent
            while parent.name not in scopes:
                if parent.name in loops:
                    outermost = parent
                parent = parent.parent
            if outermost is None:
                entry = parent.regions[0].blocks[0]
            else:
                entry = outermost.block
            alloca.operation.move_before(entry.operations[0])
        return DiagnosedSilenceableFailure.Success


_AMX_DOTS = ("x86.amx.tile_mulf", "x86.amx.tile_muli")
_AMX_TILE_REGISTERS = 8
# Ops an AMX operand load may be moved across: none writes memory.
_AMX_SCHEDULABLE = (
    *_AMX_DOTS,
    "x86.amx.tile_load",
    "affine.apply",
    "memref.subview",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


def _is_tile(value: ir.Value) -> bool:
    return str(value.type).startswith("!x86.amx.tile")


def _defining_op(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    return None if isinstance(owner, ir.Block) else _opview(owner).operation


def _tile_registers(order: list[ir.Operation]) -> int:
    """The most AMX tile registers live at once running ``order`` (one block's
    ops): a dot-product accumulates into its accumulator's register."""
    last_use: dict[ir.Value, int] = {}
    live: set[ir.Value] = set()
    for i, op in enumerate(order):
        for operand in op.operands:
            if _is_tile(operand):
                last_use[operand] = i
                if (defining := _defining_op(operand)) is None or defining not in order:
                    live.add(operand)
    most = len(live)
    for i, op in enumerate(order):
        results = [r for r in op.results if _is_tile(r) and r in last_use]
        dying = {v for v in op.operands if _is_tile(v) and last_use.get(v) == i}
        reused = 1 if op.name in _AMX_DOTS and op.operands[2] in dying else 0
        most = max(most, len(live) + len(results) - reused)
        live -= dying
        live.update(results)
    return most


def _schedule_amx_loads(block: ir.Block, distance: int) -> None:
    """Move each AMX operand tile load of ``block`` before the dot-product
    ``distance`` dot-products ahead of its first use (oneDNN's order for 1:
    the next operand loads while the current dot-product runs)."""
    order = [op.operation for op in block.operations]
    dots = [op for op in order if op.name in _AMX_DOTS]
    if len(dots) < 2:
        return
    loads = []
    for op in order:
        if op.name != "x86.amx.tile_load":
            continue
        uses = list(op.results[0].uses)
        if not uses or any(
            _opview(use.owner).operation not in dots or use.operand_number > 1
            for use in uses
        ):
            continue
        first_use = min(dots.index(_opview(use.owner).operation) for use in uses)
        loads.append((max(first_use - distance, 0), op))
    if not loads:
        return
    start = min(order.index(dots[0]), *(order.index(load) for _, load in loads))
    crossed = order[start : order.index(dots[-1]) + 1]
    if any(
        op.name not in _AMX_SCHEDULABLE and not op.name.startswith("arith.")
        for op in crossed
    ):
        return
    loads.sort(key=operator.itemgetter(0))
    moved = {load for _, load in loads}
    new_order = [op for op in order if op not in moved]
    for target, load in loads:
        new_order.insert(new_order.index(dots[target]), load)
    position = {op: i for i, op in enumerate(new_order)}
    for _, load in loads:
        for operand in load.operands:
            defining = _defining_op(operand)
            if defining in position and position[defining] > position[load]:
                return
    if _tile_registers(new_order) > _AMX_TILE_REGISTERS:
        return
    for target, load in loads:
        load.move_before(dots[target])


@_transform_op(modifies_payload=True)
class ScheduleAmxLoadsOp(HelionTransformDialect.Operation, name="schedule_amx_loads"):
    """Interleave the AMX operand tile loads of every block with its
    dot-products (see ``_schedule_amx_loads``). Blocks writing memory between
    their dot-products, or needing more tile registers so, stay as they are."""

    target: ext.Operand[transform.AnyOpType]
    distance: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)
    )

    @staticmethod
    def run(
        op: "ScheduleAmxLoadsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        blocks: list[ir.Block] = []

        def collect(visited: ir.Operation) -> ir.WalkResult:
            if visited.name in _AMX_DOTS and visited.block not in blocks:
                blocks.append(visited.block)
            return ir.WalkResult.ADVANCE

        for target in state.get_payload_ops(op.target):
            target.operation.walk(collect)
        for block in blocks:
            _schedule_amx_loads(block, op.distance.value)
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


def materialize_operand_pads(packed_only: bool = False) -> ir.Module:
    """Schedule: every function's zero-padded contraction operands in memory
    (with ``packed_only``, those read through an operand pack)."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        MaterializeOperandPadsOp(
            target=funcs,
            packed_only=ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), int(packed_only)
            ),
        )
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def version_padded_operands() -> ir.Module:
    """Schedule: branch every function's contractions on their runtime-padded
    inputs padding nothing."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        VersionPaddedOperandsOp(target=funcs)
        lh_transform.cleanup(named_seq.bodyTarget)
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


OPERAND_PACK_ATTR_NAME = "helion.operand_pack"


@_transform_op(modifies_payload=True)
class MarkOperandPacksOp(HelionTransformDialect.Operation, name="mark_operand_packs"):
    """Mark every ``_is_operand_pack`` op in the target with
    ``OPERAND_PACK_ATTR_NAME``."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MarkOperandPacksOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for linalg_op in _payload_ops(
            state, op.target, (linalg.TransposeOp, linalg.GenericOp)
        ):
            if _is_operand_pack(linalg_op):
                linalg_op.operation.attributes[OPERAND_PACK_ATTR_NAME] = (
                    ir.UnitAttr.get()
                )
        return DiagnosedSilenceableFailure.Success


def isolate_operand_packs() -> ir.Module:
    """Schedule: tile every operand pack by 1 on its outer dim.

    A loop result is no producer tile-and-fuse can fuse: the pack stays outside
    the contraction's register loops and runs once per contraction, not once
    per register tile reading it.
    """
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        MarkOperandPacksOp(target=funcs)
        packs = structured.MatchOp(
            transform.any_op_t(),
            named_seq.bodyTarget,
            op_attrs=ir.DictAttr.get({OPERAND_PACK_ATTR_NAME: ir.UnitAttr.get()}),
        )
        with lh_transform.foreach(packs) as pack:
            structured.TileUsingForOp(pack, sizes=[1])
            transform.yield_()
        transform.yield_()
    return schedule


def hoist_allocas() -> ir.Module:
    """Schedule: hoist every function's static stack buffers out of loops."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        HoistAllocasOp(target=funcs)
        transform.yield_()
    return schedule


def fold_empty_slices() -> ir.Module:
    """Schedule: slices of empty tensors as empty tensors of the slice shape.

    A per-register-tile op writing a slice of a whole-tile temporary (e.g. a
    fused accumulator or epilogue) then gets a register-tile-sized buffer, not
    one the size of the whole tile streaming through the caches.
    """
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            transform_tensor.apply_patterns_tensor_fold_tensor_empty()
            transform.apply_patterns_canonicalization()
        transform.yield_()
    return schedule


def schedule_amx_loads(distance: int = 1) -> ir.Module:
    """Schedule: interleave every function's AMX operand loads with its
    dot-products, each ``distance`` dot-products ahead of its first use."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ScheduleAmxLoadsOp(
            target=funcs,
            distance=ir.IntegerAttr.get(ir.IntegerType.get_signless(64), distance),
        )
        transform.yield_()
    return schedule


def promote_buffers_to_stack(max_alloc_size_in_bytes: int = 262144) -> ir.Module:
    """Schedule: ``promote-buffers-to-stack`` on every function, with a size limit
    (the pipeline descriptor cannot nest a pass with options)."""
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        transform.apply_registered_pass(
            transform.AnyOpType.get(),
            funcs,
            "promote-buffers-to-stack",
            options={"max-alloc-size-in-bytes": max_alloc_size_in_bytes},
        )
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
