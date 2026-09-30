"""Python-defined transform ops of the opt pipeline and the schedules applying them.

The ops share a module because a Python-defined dialect only registers the ops
defined before it is loaded.

``vectorize_pads``: bufferized, a pad is a temporary zeroed and then copied
into, and vectorize_all's copy forwarding (upstream
``LinalgCopyVTRForwardingPattern``) reads past the copy with poison padding. As
a vector read of the pad's source with the pad value past its end, written into
an empty tensor, there is no such copy.

``split_transfers``: tile extents are runtime values, so every tile's transfer
may be out of bounds and lowers to masked accesses. Upstream's full/partial
split guards each with an in-bounds check, so only edge tiles take the masked
path. Upstream loops forever on rank-reducing transfers, so these are first
given the memref's rank.

``vectorize_linalg``: lighthouse vectorizes without vector sizes, which fails
for ops of runtime shape. Tiling bounds each runtime extent (an ``affine.min``
with a constant), so such ops are vectorized with masks, the bounds as vector
sizes. Masked contractions are unmasked for upstream's x86 contraction patterns.
"""

from collections.abc import Sequence
import math

from lighthouse.dialects import DialectExtension
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import affine
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector as transform_vector


class HelionTransformDialect(DialectExtension, name="helion_transform"):
    """Transform ops of the Helion MLIR backend's pipelines."""


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


class VectorizePadsOp(HelionTransformDialect.Operation, name="vectorize_pads"):
    """Vectorize every ``tensor.pad`` in the target with no low padding, a
    constant padding value and evident bounds of its runtime extents."""

    target: ext.Operand[transform.AnyOpType]

    @classmethod
    def attach_interface_impls(cls, context: ir.Context | None = None) -> None:
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "VectorizePadsOp",
            rewriter: transform.TransformRewriter,
            _results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            pads: list[tensor.PadOp] = []

            def collect(visited: ir.Operation) -> ir.WalkResult:
                if isinstance(visited.opview, tensor.PadOp):
                    pads.append(visited.opview)
                return ir.WalkResult.ADVANCE

            for target in state.get_payload_ops(op.target):
                target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
            for pad in pads:
                _vectorize(pad, rewriter)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "VectorizePadsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "VectorizePadsOp") -> list:
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.modifies_payload()
            )


def _expand_rank(
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
    if (
        not 0 < vector_rank < rank
        or all(in_bounds)
        or transfer.permutation_map.value != minor_identity
    ):
        return
    unit_dims = rank - vector_rank
    full_type = ir.VectorType.get(
        [1] * unit_dims + vector_type.shape, vector_type.element_type
    )
    identity = ir.AffineMap.get_minor_identity(rank, rank)
    # Dims outside the transfer are in bounds by definition.
    full_in_bounds = [True] * unit_dims + in_bounds
    with ir.InsertionPoint(transfer), transfer.location:
        if is_read:
            read = vector.TransferReadOp(
                full_type,
                transfer.base,
                transfer.indices,
                identity,
                transfer.padding,
                full_in_bounds,
            ).result
            rewriter.replace_op(
                transfer, [vector.ShapeCastOp(vector_type, read).result]
            )
        else:
            value = vector.ShapeCastOp(full_type, transfer.valueToStore).result
            vector.TransferWriteOp(
                None, value, transfer.base, transfer.indices, identity, full_in_bounds
            )
            rewriter.erase_op(transfer)


class ExpandTransferRankOp(
    HelionTransformDialect.Operation, name="expand_transfer_rank"
):
    """Give every possibly out-of-bounds, minor-identity, unmasked memref transfer
    in the target the memref's rank, through leading unit dims."""

    target: ext.Operand[transform.AnyOpType]

    @classmethod
    def attach_interface_impls(cls, context: ir.Context | None = None) -> None:
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ExpandTransferRankOp",
            rewriter: transform.TransformRewriter,
            _results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            transfers: list[vector.TransferReadOp | vector.TransferWriteOp] = []

            def collect(visited: ir.Operation) -> ir.WalkResult:
                if isinstance(
                    visited.opview, (vector.TransferReadOp, vector.TransferWriteOp)
                ):
                    transfers.append(visited.opview)
                return ir.WalkResult.ADVANCE

            for target in state.get_payload_ops(op.target):
                target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
            for transfer in transfers:
                _expand_rank(transfer, rewriter)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ExpandTransferRankOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "ExpandTransferRankOp") -> list:
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.modifies_payload()
            )


def _constant(value: ir.Value) -> int | None:
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
    if (constant := _constant(value)) is not None:
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
    if isinstance(op, tensor.DimOp) and (dim := _constant(op.index)) is not None:
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


def _runtime_tile_sizes(op: ir.OpView) -> list[int] | None:
    """Tile sizes making a linalg op's loop ranges bounded, their bounds multiples
    of ``_RUNTIME_TILE`` and its masked vectors at most ``_MAX_VECTOR_ELEMENTS``
    (0: untiled)."""
    bounds = _loop_bounds(op)
    if bounds is None:
        return None
    too_large = None not in bounds and math.prod(bounds) > _MAX_VECTOR_ELEMENTS
    return [
        _RUNTIME_TILE
        if bound is None
        or (bound > _RUNTIME_TILE and (too_large or bound % _RUNTIME_TILE))
        else 0
        for bound in bounds
    ]


# Loop counts of the ops vectorize_linalg vectorizes with masks.
_MAX_LOOPS = 8
# Tile size of runtime extents without an evident bound, of odd bounds, and of all
# extents of ops whose masked vectors would exceed _MAX_VECTOR_ELEMENTS: LLVM takes
# many seconds on large or odd-width masked vectors.
_RUNTIME_TILE = 32
_MAX_VECTOR_ELEMENTS = 4096


class PartitionLinalgOp(HelionTransformDialect.Operation, name="partition_linalg"):
    """Group the target's linalg ops: those of static shape; those of runtime
    shape with evident loop bounds, by loop count (1 to ``_MAX_LOOPS``); and
    those needing runtime tiling first (``_runtime_tile_sizes``)."""

    target: ext.Operand[transform.AnyOpType]
    groups: Sequence[ext.Result[transform.AnyOpType]]

    @classmethod
    def attach_interface_impls(cls, context: ir.Context | None = None) -> None:
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "PartitionLinalgOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            groups: list[list[ir.Operation]] = [[] for _ in op.groups]

            def collect(visited: ir.Operation) -> ir.WalkResult:
                bounds = _loop_bounds(visited.opview)
                if bounds is None:
                    return ir.WalkResult.ADVANCE
                shaped = [
                    ir.ShapedType(value.type)
                    for value in [*visited.operands, *visited.results]
                    if isinstance(value.type, ir.ShapedType)
                ]
                if all(value.has_static_shape for value in shaped):
                    groups[0].append(visited)
                elif any(_runtime_tile_sizes(visited.opview)):
                    groups[-1].append(visited)
                elif 0 < len(bounds) < len(groups) - 1:
                    groups[len(bounds)].append(visited)
                return ir.WalkResult.ADVANCE

            for target in state.get_payload_ops(op.target):
                target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
            for handle, ops in zip(op.groups, groups, strict=True):
                results.set_ops(handle, ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "PartitionLinalgOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "PartitionLinalgOp") -> list:
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


class LoopBoundsOp(HelionTransformDialect.Operation, name="loop_bounds"):
    """The constant upper bounds of one linalg op's loop ranges, one param each."""

    target: ext.Operand[transform.AnyOpType]
    bounds: Sequence[ext.Result[transform.AnyParamType]]

    @classmethod
    def attach_interface_impls(cls, context: ir.Context | None = None) -> None:
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
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

        @staticmethod
        def allow_repeated_handle_operands(_op: "LoopBoundsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "LoopBoundsOp") -> list:
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


class RuntimeTileSizesOp(HelionTransformDialect.Operation, name="runtime_tile_sizes"):
    """The ``_runtime_tile_sizes`` of one linalg op, as one param."""

    target: ext.Operand[transform.AnyOpType]
    sizes: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context: ir.Context | None = None) -> None:
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "RuntimeTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            sizes = (
                _runtime_tile_sizes(targets[0].opview) if len(targets) == 1 else None
            )
            if sizes is None:
                return DiagnosedSilenceableFailure.SilenceableFailure
            i64 = ir.IntegerType.get_signless(64)
            results.set_params(op.sizes, [ir.IntegerAttr.get(i64, s) for s in sizes])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "RuntimeTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "RuntimeTileSizesOp") -> list:
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


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


class UnmaskContractionsOp(
    HelionTransformDialect.Operation, name="unmask_contractions"
):
    """Rewrite every masked floating-point add-contraction in the target as an
    unmasked one of operands zeroed where masked off."""

    target: ext.Operand[transform.AnyOpType]

    @classmethod
    def attach_interface_impls(cls, context: ir.Context | None = None) -> None:
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "UnmaskContractionsOp",
            rewriter: transform.TransformRewriter,
            _results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            masks: list[vector.MaskOp] = []

            def collect(visited: ir.Operation) -> ir.WalkResult:
                if isinstance(visited.opview, vector.MaskOp):
                    masks.append(visited.opview)
                return ir.WalkResult.ADVANCE

            for target in state.get_payload_ops(op.target):
                target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
            for masked in masks:
                _unmask_contraction(masked, rewriter)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "UnmaskContractionsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "UnmaskContractionsOp") -> list:
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.modifies_payload()
            )


def vectorize_linalg() -> ir.Module:
    """Schedule: lighthouse's ``vectorization.py[gen=vectorize_linalg]``, also
    vectorizing ops of runtime shape, with masks. Their runtime extents without
    an evident bound are first tiled; ops that cannot be are left to the loop
    lowering."""
    HelionTransformDialect.load()
    groups = [transform.AnyOpType.get()] * (_MAX_LOOPS + 2)
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        unbounded = PartitionLinalgOp(target=funcs, groups=groups).groups[-1]
        with lh_transform.foreach(unbounded) as op:
            sizes = RuntimeTileSizesOp(target=op).sizes
            # Tiling fails, changing nothing, on some ops (e.g. keepdim reductions).
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Suppress, [], op
            )
            with ir.InsertionPoint(sequence.body):
                structured.TileUsingForOp(sequence.bodyTarget, sizes=sizes)
                transform.yield_()
            transform.yield_()
        static, *dynamic, _ = PartitionLinalgOp(target=funcs, groups=groups).groups
        with lh_transform.foreach(static) as op:
            structured.structured_vectorize(op, [], create_named_contraction=True)
            transform.yield_()
        for loops, group in enumerate(dynamic, 1):
            with lh_transform.foreach(group) as op:
                # Ops masked vectorization rejects (e.g. argmax, gathers) stay for
                # the loop lowering.
                sequence = transform.SequenceOp(
                    transform.FailurePropagationMode.Suppress, [], op
                )
                with ir.InsertionPoint(sequence.body):
                    bounds = LoopBoundsOp(
                        target=sequence.bodyTarget,
                        bounds=[transform.AnyParamType.get()] * loops,
                    ).bounds
                    structured.structured_vectorize(
                        sequence.bodyTarget,
                        list(bounds),
                        static_vector_sizes=[ir.ShapedType.get_dynamic_size()] * loops,
                        scalable_sizes=[False] * loops,
                        create_named_contraction=True,
                    )
                    transform.yield_()
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


def split_transfers() -> ir.Module:
    """Schedule: full/partial split of every function's out-of-bounds memref
    transfers. Needs an allocation scope around each transfer."""
    HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ExpandTransferRankOp(target=funcs)
        with ir.InsertionPoint(transform.ApplyPatternsOp(funcs).patterns):
            transform_vector.apply_patterns_vector_split_transfer_full_partial(
                split_transfer_strategy=transform_vector.VectorTransferSplit.VectorTransfer
            )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
