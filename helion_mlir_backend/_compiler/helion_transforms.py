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
"""

from lighthouse.dialects import DialectExtension
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure
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
    if not result_type.has_static_shape or list(pad.low) or any(pad.static_low):
        return
    with ir.InsertionPoint(pad), pad.location:
        padding = _pad_value(pad)
        if padding is None:
            return
        shape = result_type.shape
        source_shape = ir.RankedTensorType(pad.source.type).shape
        identity = ir.AffineMap.get_minor_identity(len(shape), len(shape))
        origin = [arith.ConstantOp(ir.IndexType.get(), 0).result] * len(shape)
        read = vector.TransferReadOp(
            ir.VectorType.get(shape, result_type.element_type),
            pad.source,
            origin,
            identity,
            padding,
            [real == padded for real, padded in zip(source_shape, shape, strict=True)],
        ).result
        empty = tensor.EmptyOp(shape, result_type.element_type).result
        written = vector.TransferWriteOp(
            result_type, read, empty, origin, identity, [True] * len(shape)
        ).result
    rewriter.replace_op(pad, [written])


class VectorizePadsOp(HelionTransformDialect.Operation, name="vectorize_pads"):
    """Vectorize every statically shaped ``tensor.pad`` in the target with no low
    padding and a constant padding value."""

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
