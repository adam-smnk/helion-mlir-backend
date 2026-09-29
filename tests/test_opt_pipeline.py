"""End-to-end execution under the backend's optimizing lighthouse pipeline.

f32 only: bf16 needs AMX hardware to execute (covered at IR level by test_amx_ir_gate.py).
Tiles stay >= 32 because lighthouse's tile-and-fuse aborts when every tiled dim of an op
is smaller than its cache tile (see docs/MLIR_LIMITATIONS.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import opt_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.isolated, pytest.mark.slow]


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32])
)
def opt_elementwise_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.relu(x[tm, tn] * 2.0 + y[tm, tn])
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32, 32])
)
def opt_addmm_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 128])
)
def opt_row_sum_loop_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 128])
)
def opt_row_max_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.full([tm], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(n):
            acc = torch.maximum(acc, x[tm, tn].amax(dim=-1))
        out[tm] = acc
    return out


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 32, 32, 32]),
)
def opt_ragged_k_bmm_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([batch, m, n], dtype=x.dtype, device=x.device)
    for tb, tm, tn in hl.tile([batch, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.baddbmm(acc, x[tb, tm, tk], y[tb, tk, tn])
        out[tb, tm, tn] = acc
    return out


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[32, 128, 128]),
)
def opt_online_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
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


def _cpu_utils_cases() -> dict[str, tuple[Callable[[], object], Callable[[], object]]]:
    import helion_mlir_cpu_utils as cpu

    a, b = torch.randn(256, 192), torch.randn(192, 320)
    bias = torch.randn(320)
    batch_a, batch_b = torch.randn(2, 64, 96), torch.randn(2, 96, 128)
    layer = torch.nn.Linear(192, 160)
    rows, ragged_rows = torch.randn(64, 1024), torch.randn(64, 1000)
    ragged_a, ragged_b = torch.randn(2, 32, 40), torch.randn(2, 40, 32)
    return {
        "matmul": (lambda: cpu.matmul(a, b), lambda: a @ b),
        "matmul_bias_relu": (
            lambda: cpu.matmul(a, b, bias=bias, epilogue=torch.relu),
            lambda: torch.relu(a @ b + bias),
        ),
        "matmul_trans_b": (
            lambda: cpu.matmul(a, b.t().contiguous(), trans_b=True),
            lambda: a @ b,
        ),
        "bmm": (lambda: cpu.bmm(batch_a, batch_b), lambda: batch_a @ batch_b),
        "linear": (lambda: cpu.linear(a, layer)[0], lambda: layer(a).detach()),
        "elementwise": (
            lambda: opt_elementwise_kernel(
                a[:, :64].contiguous(), a[:, 64:128].contiguous()
            ),
            lambda: torch.relu(a[:, :64] * 2.0 + a[:, 64:128]),
        ),
        "addmm": (
            lambda: opt_addmm_kernel(a[:64], b[:, :64].contiguous()),
            lambda: a[:64] @ b[:, :64],
        ),
        "row_sum_loop": (lambda: opt_row_sum_loop_kernel(rows), lambda: rows.sum(-1)),
        "row_max_ragged": (
            lambda: opt_row_max_kernel(ragged_rows),
            lambda: ragged_rows.amax(-1),
        ),
        "online_softmax": (
            lambda: opt_online_softmax_kernel(ragged_rows),
            lambda: ragged_rows.softmax(-1),
        ),
        # Padded batch_matmul operands take the scalar pipeline (opt miscompiles them).
        "ragged_k_bmm": (
            lambda: opt_ragged_k_bmm_kernel(ragged_a, ragged_b),
            lambda: ragged_a @ ragged_b,
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "matmul",
        "matmul_bias_relu",
        "matmul_trans_b",
        "bmm",
        "linear",
        "elementwise",
        "addmm",
        "row_sum_loop",
        "row_max_ragged",
        "online_softmax",
        "ragged_k_bmm",
    ],
)
def test_optimizing_pipeline_f32(case: str) -> None:
    torch.manual_seed(0)
    run, reference = _cpu_utils_cases()[case]
    with opt_pipeline():
        actual = run()
    torch.testing.assert_close(actual, reference(), atol=1e-3, rtol=1e-3)
