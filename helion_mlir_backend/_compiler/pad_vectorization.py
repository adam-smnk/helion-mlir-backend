"""Tensor-level vectorization of ``tensor.pad``.

Bufferized, a pad is a temporary zeroed and then copied into, and vectorize_all's
copy forwarding (upstream ``LinalgCopyVTRForwardingPattern``) reads past the
copy with poison padding. As a vector read of the pad's source with the pad value
past its end, written into an empty tensor, there is no such copy.
"""

from __future__ import annotations

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
            op: VectorizePadsOp,
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
        def allow_repeated_handle_operands(_op: VectorizePadsOp) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: VectorizePadsOp) -> list:
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
