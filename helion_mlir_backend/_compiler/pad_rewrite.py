"""Pre-bufferization rewrites of ``tensor.pad``."""

from __future__ import annotations

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def pads_to_fill() -> ir.Module:
    """Fold pads into the vector reads of them and whole-pad inserts (a masked
    read of the source), then rewrite the rest as ``tensor.empty`` +
    ``linalg.fill`` + ``tensor.insert_slice``.

    Bufferized as is, a pad becomes a zeroed allocation that vectorize_all's copy
    forwarding reads past with poison padding; a ``linalg.fill`` alone does not
    prevent that, since vectorize_all vectorizes the fill first.
    """
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            structured.apply_patterns_linalg_pad_vectorization()
        lh_transform.cleanup(named_seq.bodyTarget)
        pads = lh_transform.match_op(named_seq.bodyTarget, "tensor.pad")
        structured.structured_rewrite_in_destination_passing_style(
            transform.any_op_t(), pads
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
