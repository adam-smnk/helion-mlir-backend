"""Loop-carried elementwise updates write into their iter arg (``mlir/in_place.py``)."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.execution import inline_module


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16))
def row_sum_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16, 16))
def online_softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        mi = hl.full([tm], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            values = x[tm, tn]
            mi_next = torch.maximum(mi, torch.amax(values, dim=1))
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tn in hl.tile(n):
            out[tm, tn] = torch.exp(x[tm, tn] - mi[:, None]) / di[:, None]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 16))
def scaled_running_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = torch.maximum(acc * 0.5, x[tm, tn].amax(dim=-1))
        out[tm] = acc
    return out


def _scaled_running_max_reference(x: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(x.size(0))
    for start in range(0, x.size(1), 16):
        acc = torch.maximum(acc * 0.5, x[:, start : start + 16].amax(dim=-1))
    return acc


def _writes_into_iter_arg(kernel: helion.Kernel, x: torch.Tensor) -> list[bool]:
    """For each loop-carried value updated by a ``linalg.generic``, whether the
    update's destination is the loop's iter arg."""
    module = inline_module(generate_mlir(kernel, [x]))
    found = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.for":
            body = op.regions[0].blocks[0]
            yielded = list(body.operations)[-1].operands
            for arg, value in zip(list(body.arguments)[1:], yielded, strict=True):
                producer = (
                    value.owner.operation if isinstance(value, ir.OpResult) else None
                )
                if producer is not None and producer.name == "linalg.generic":
                    found.append(producer.opview.outputs[value.result_number] == arg)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def test_accumulation_writes_into_its_iter_arg() -> None:
    assert _writes_into_iter_arg(row_sum_loop, torch.randn(16, 64)) == [True]


def test_accumulated_reduction_starts_from_its_iter_arg() -> None:
    # ``acc + x.sum(-1)``: the sum reduces into ``acc``, no separate add is left.
    module = inline_module(generate_mlir(row_sum_loop, [torch.randn(16, 64)]))
    loops = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "scf.for":
            loops.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    (loop,) = loops
    body = loop.regions[0].blocks[0]
    (value,) = list(body.operations)[-1].operands
    producer = value.owner.operation
    assert "reduction" in str(producer.attributes["iterator_types"])
    assert producer.opview.outputs[0] == list(body.arguments)[1]
    generics = [op for op in body.operations if op.name == "linalg.generic"]
    assert len(generics) == 1


def test_update_reading_the_old_value_later_is_still_in_place() -> None:
    # ``mi`` is read after ``mi_next`` is computed: bufferization copies it.
    # ``di``'s update is ``add(mul(di, ...), ...)``: the add does not read ``di``.
    found = _writes_into_iter_arg(online_softmax, torch.randn(16, 64))
    assert sorted(found) == [False, True]


def test_update_not_reading_the_iter_arg_keeps_its_destination() -> None:
    assert _writes_into_iter_arg(scaled_running_max, torch.randn(16, 64)) == [False]


@pytest.mark.parametrize(
    ("kernel", "reference"),
    [
        (row_sum_loop, lambda x: x.sum(-1)),
        (online_softmax, lambda x: x.softmax(-1)),
        (scaled_running_max, _scaled_running_max_reference),
    ],
    ids=["row_sum_loop", "online_softmax", "scaled_running_max"],
)
def test_in_place_updates_execute(kernel, reference) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        reference,
        [torch.randn(16, 60)],
        paths=("direct", "generated"),
        atol=1e-5,
        rtol=1e-5,
    )
