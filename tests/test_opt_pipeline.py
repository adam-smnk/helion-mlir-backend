"""End-to-end execution under the backend's optimizing lighthouse pipeline.

f32 only: bf16 needs AMX hardware to execute (covered at IR level by test_amx_ir_gate.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests.harness import opt_pipeline

from helion_mlir_backend._compiler.helion_transforms import hoist_allocas
from helion_mlir_backend._compiler.helion_transforms import legalize_for_llvm
from helion_mlir_backend._compiler.helion_transforms import split_transfers

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
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32])
)
def opt_col_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    k, n = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tn in hl.tile(n):
        acc = hl.zeros([tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc + x[tk, tn].sum(dim=0)
        out[tn] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32])
)
def opt_tile_sums_kernel(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    out = torch.ones([(n + 31) // 32, x.size(1)], dtype=x.dtype, device=x.device)
    for t in hl.tile(n):
        out[t.id, :] = out[t.id, :] + x[t, :].sum(0)
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


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[2, 4, 4])
)
def opt_packed_gemm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pack A and B into 32x32 blocks, then a blocked contraction over tiles of
    blocks (partial ones too)."""
    m, k = a.size()
    _, n = b.size()
    mb, kb, nb = (m + 31) // 32, (k + 31) // 32, (n + 31) // 32
    a4 = torch.empty((mb, kb, 32, 32), dtype=a.dtype, device=a.device)
    b4 = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    out4 = torch.empty((mb, 32, nb, 32), dtype=a.dtype, device=a.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        a4[tm.id, tk.id, :, :] = a[tm, tk]
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        b4[tn.id, tk.id, :, :] = b[tk, tn]
    hl.barrier()
    for tbm, tbn in hl.tile([mb, nb]):
        acc = hl.zeros([tbm, tbn, 32, 32], dtype=torch.float32)
        for tbk in hl.tile(kb):
            acc = acc + torch.einsum(
                "akmc,bkcn->abmn", a4[tbm, tbk, :, :], b4[tbn, tbk, :, :]
            )
        out4[tbm, :, tbn, :] = acc.permute(0, 2, 1, 3)
    return out4.view(mb * 32, nb * 32)[:m, :n]


def _cpu_utils_cases() -> dict[str, tuple[Callable[[], object], Callable[[], object]]]:
    import helion_mlir_cpu_utils as cpu

    a, b = torch.randn(256, 192), torch.randn(192, 320)
    bias = torch.randn(320)
    batch_a, batch_b = torch.randn(2, 64, 96), torch.randn(2, 96, 128)
    layer = torch.nn.Linear(192, 160)
    rows, ragged_rows = torch.randn(64, 1024), torch.randn(64, 1000)
    ragged_a, ragged_b = torch.randn(2, 32, 40), torch.randn(2, 40, 32)
    ragged_cols = torch.randn(70, 64)
    gemm_a, gemm_b = torch.randn(70, 200), torch.randn(200, 150)
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
        # Padded batch_matmul operands (miscompiled to NaNs without vectorize_pads).
        "ragged_k_bmm": (
            lambda: opt_ragged_k_bmm_kernel(ragged_a, ragged_b),
            lambda: ragged_a @ ragged_b,
        ),
        # Reductions over padded tiles (wrong without vectorize_pads).
        "col_sum_ragged": (
            lambda: opt_col_sum_kernel(ragged_cols),
            lambda: ragged_cols.sum(0),
        ),
        "tile_sums_ragged": (
            lambda: opt_tile_sums_kernel(ragged_cols),
            lambda: (
                1
                + torch.nn.functional.pad(ragged_cols, (0, 0, 0, 26))
                .reshape(-1, 32, 64)
                .sum(1)
            ),
        ),
        "packed_gemm_ragged": (
            lambda: opt_packed_gemm_kernel(gemm_a, gemm_b),
            lambda: gemm_a @ gemm_b,
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
        "col_sum_ragged",
        "tile_sums_ragged",
        "packed_gemm_ragged",
    ],
)
def test_optimizing_pipeline_f32(case: str) -> None:
    torch.manual_seed(0)
    run, reference = _cpu_utils_cases()[case]
    with opt_pipeline():
        actual = run()
    torch.testing.assert_close(actual, reference(), atol=1e-3, rtol=1e-3)


RANK_REDUCING_TRANSFERS = """
func.func @f(%m: memref<?x?xf32>, %i: index, %v: vector<32xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %read = vector.transfer_read %m[%i, %c0], %pad : memref<?x?xf32>, vector<32xf32>
  vector.transfer_write %v, %m[%i, %c0] : vector<32xf32>, memref<?x?xf32>
  return %read : vector<32xf32>
}
"""


def _apply(make_schedule: Callable[[], ir.Module], source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        schedule = make_schedule()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def test_split_transfers_rank_reducing() -> None:
    text = _apply(split_transfers, RANK_REDUCING_TRANSFERS)
    # The read and the write, each on its in-bounds path (cleanup merges the ifs).
    assert "scf.if" in text
    assert text.count("in_bounds = [true]") == 2


ZERO_D_TRANSFERS = """
func.func @f(%m: memref<1x1xf32, strided<[?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %read = vector.transfer_read %m[%c0, %c0], %pad : memref<1x1xf32, strided<[?, ?], offset: ?>>, vector<f32>
  vector.transfer_write %read, %m[%c0, %c0] : vector<f32>, memref<1x1xf32, strided<[?, ?], offset: ?>>
  return
}
"""


def test_legalize_0d_transfers() -> None:
    text = _apply(legalize_for_llvm, ZERO_D_TRANSFERS)
    assert "vector.transfer" not in text
    assert "memref.load" in text
    assert "memref.store" in text


ALLOCA_IN_LOOP = """
func.func @f(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %buffer = memref.alloca() : memref<f32>
    %value = memref.load %buffer[] : memref<f32>
    memref.store %value, %buffer[] : memref<f32>
  }
  return
}
"""


def test_hoist_allocas_out_of_loops() -> None:
    text = _apply(hoist_allocas, ALLOCA_IN_LOOP)
    assert text.index("memref.alloca") < text.index("scf.for")
