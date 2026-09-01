"""Typed per-loop-level synthetic store accumulator context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir

    from .slice_plan import SlicePlan


@dataclass
class ForStoreContext:
    """Tracks a synthetic per-iteration accumulator threaded through nested
    ``scf.for`` loops between the outer ``scf.forall`` and the store that
    eventually flushes into it.

    ``current`` is the live accumulator value for the active loop iteration,
    rebound each time a new value is produced (e.g. after an insert_slice).
    ``store_plan`` is computed once, on first store, and reused thereafter.
    ``flush_offsets`` are the offsets this level's finished accumulator is
    inserted at when it flushes into its parent (or the outer forall).
    """

    flush_offsets: list[ir.Value]
    current: ir.Value | None = None
    store_plan: SlicePlan | None = None
    # id() of the destination FakeTensor this eventually flushes into (when
    # known) -- routes the flush to the right output when a phase has >1.
    target_tensor_id: int | None = None
