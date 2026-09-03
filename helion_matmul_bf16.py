"""4K bf16 matmul benchmark for Helion's MLIR backend.

Tuned for a 2-socket Intel Xeon Platinum 8592+ (Emerald Rapids, 64 cores and
256 MiB of private L2 per socket) running one thread per physical core on a
single socket:

    HELION_MLIR_PIPELINE=1 OMP_NUM_THREADS=64 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    LD_PRELOAD=/lib64/libtcmalloc.so:$LD_PRELOAD \
    PYTHONPATH=~/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH \
    uv run python helion_matmul_bf16.py

The packed-RHS path packs B into column panels with `helion_block_pack`'s
validated `pack_b_panels`, then runs one plain 2D matmul call per panel from
ordinary (eager, untraced) Python. Writing each panel's `[M, BN]` result
directly into a slice of a flat `[M, N]` output tensor avoids the rank-4
blocked-output store that the MLIR backend cannot lower: keeping the panel
loop outside the traced kernel sidesteps that entirely, at the cost of one
kernel launch per panel. Register blocking, VNNI packing and AMX tile
selection inside each panel matmul are done by the lighthouse pipeline, which
recognises a `linalg.matmul ins(bf16, bf16) outs(f32)` contraction on an
amx_tile target and lowers it to `tdpbf16ps`.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_block_pack
import helion_mlir_backend  # noqa: F401

# Helion probes tile shapes during tracing and torch logs the recovered failures.
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)

if TYPE_CHECKING:
    from collections.abc import Callable


SIZE = 4096
TILE_M = int(os.environ.get("HELION_MATMUL_TILE_M", "128"))
TILE_N = int(os.environ.get("HELION_MATMUL_TILE_N", "1024"))
TILE_K = int(os.environ.get("HELION_MATMUL_TILE_K", "32"))
# Panel width for the packed path: each panel is one host-side matmul call, so
# this also sets the per-panel N tile. 256 was the sweep optimum (128/512/1024
# were all slower) at 4096x4096, 4 threads.
PACK_BN = int(os.environ.get("HELION_MATMUL_PACK_BN", "256"))
PACK_TILE_M = int(os.environ.get("HELION_MATMUL_PACK_TILE_M", "128"))
PACK_TILE_K = int(os.environ.get("HELION_MATMUL_PACK_TILE_K", "32"))
WARMUP_ITERS = 5
BENCHMARK_ITERS = 20
SAMPLES = 7

# The pipeline's bf16 register tiles are 32x32 parallel by 32 deep. A deeper K
# cache tile turns the register-level reduction into a loop with vector-typed
# iter_args, which the AMX conversion cannot handle: `vector.contract` then
# survives to LLVM translation and the ExecutionEngine fails to build.
AMX_PARALLEL_TILE = 32
AMX_REDUCTION_TILE = 32


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[TILE_M, TILE_N, TILE_K]),
)
def matmul_bf16_mlir(x: Tensor, y: Tensor) -> Tensor:
    """Compute a bf16 matrix product with f32 accumulation over L2-resident tiles.

    The K loop is sequential and steps by exactly one AMX reduction tile, which
    is what lets the pipeline turn the contraction into a chain of `tdpbf16ps`
    updates. That also means the f32 accumulator is re-read and re-written from
    cache every 32 elements of K, so wide-N tiles win: they maximise reuse of
    each freshly loaded B panel against that fixed accumulator traffic. 128x1024
    keeps the accumulator at 512 KiB and still yields 128 tiles, two per thread
    at 64 threads.

    The result stays f32: that is the native AMX accumulator type, and the bf16
    epilogue path still fails JIT for this AMX loop shape.
    """
    m, k = x.shape
    k2, n = y.shape
    assert k == k2, "matmul dimension mismatch"

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[PACK_TILE_M, PACK_BN, PACK_TILE_K]),
)
def matmul_bf16_panel_mlir(a: Tensor, b_panel: Tensor) -> Tensor:
    """Plain 2D ``A @ B_panel`` for one packed column panel, ``[M, K] x [K, BN]``.

    Identical shape to `matmul_bf16_mlir` other than a fixed ``BN``-wide N tile;
    kept as its own kernel so a distinct config can pin the panel width.
    """
    m, k = a.shape
    k2, bn = b_panel.shape
    assert k == k2, "matmul dimension mismatch"

    out = torch.empty((m, bn), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, bn]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b_panel[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def matmul_bf16_flat_packed(a: Tensor, packed_b: Tensor) -> Tensor:
    """Row-major ``[M, N]`` from ``A`` and panel-packed ``B`` (``[N/BN, K, BN]``).

    The panel loop runs in plain eager Python, not inside a traced kernel: that
    is what lets each panel's result land straight in a slice of a flat 2D
    output, instead of a blocked ``[M/BM, N/BN, BM, BN]`` tensor whose unblocking
    store the MLIR backend cannot lower (see the AMX gaps doc for the crash).
    """
    m, k = a.shape
    panels_n, k2, bn = packed_b.shape
    assert k == k2, "matmul dimension mismatch"

    out = torch.empty((m, panels_n * bn), dtype=torch.float32, device=a.device)
    for panel in range(panels_n):
        out[:, panel * bn : (panel + 1) * bn] = matmul_bf16_panel_mlir(
            a, packed_b[panel]
        )
    return out



def benchmark(name: str, operation: Callable[[], object]) -> float:
    """Return the median per-call time in milliseconds for a synchronous CPU op."""
    for _ in range(WARMUP_ITERS):
        operation()

    timings_ms = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(BENCHMARK_ITERS):
            operation()
        timings_ms.append((time.perf_counter() - start) * 1_000 / BENCHMARK_ITERS)

    median_ms = statistics.median(timings_ms)
    print(f"{name:20s} {median_ms:8.3f} ms")
    return median_ms


def check_numerics(name: str, actual: Tensor, reference_f32: Tensor) -> None:
    """Compare against an f32 reference using bf16-appropriate tolerances."""
    actual_f32 = actual.to(torch.float32)
    abs_err = (actual_f32 - reference_f32).abs()
    rel_err = abs_err / reference_f32.abs().clamp_min(1e-6)
    print(
        f"{name:20s} max abs {abs_err.max().item():.3e}, "
        f"mean abs {abs_err.mean().item():.3e}, "
        f"max rel {rel_err.max().item():.3e}"
    )
    # bf16 has 8 mantissa bits, so a single rounding is ~2^-8; allow a few ulp
    # for the different summation order plus an absolute floor for near-zero
    # entries of a K=4096 random-normal reduction.
    torch.testing.assert_close(actual_f32, reference_f32, rtol=3e-2, atol=1.0)


def main() -> None:
    if os.environ.get("HELION_MLIR_PIPELINE") != "1":
        raise RuntimeError("Set HELION_MLIR_PIPELINE=1 to use the vectorizing pipeline")
    if any(tile % AMX_PARALLEL_TILE for tile in (TILE_M, TILE_N)):
        raise ValueError(
            f"TILE_M/TILE_N must be multiples of {AMX_PARALLEL_TILE} for AMX bf16"
        )
    if TILE_K != AMX_REDUCTION_TILE:
        raise ValueError(
            f"TILE_K must be {AMX_REDUCTION_TILE}; deeper K tiles leave a"
            " vector.contract unlowered and the JIT fails"
        )
    if PACK_TILE_K != AMX_REDUCTION_TILE:
        raise ValueError(f"PACK_TILE_K must be {AMX_REDUCTION_TILE} for AMX bf16")
    if any(SIZE % tile for tile in (TILE_M, TILE_N, TILE_K)):
        raise ValueError(
            f"Tile sizes must divide {SIZE} for this fixed-shape benchmark"
        )
    if any(SIZE % tile for tile in (PACK_TILE_M, PACK_BN, PACK_TILE_K)):
        raise ValueError(
            f"Packed tile sizes must divide {SIZE} for this fixed-shape benchmark"
        )

    threads = int(os.environ.get("OMP_NUM_THREADS", "64"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    a = torch.randn((SIZE, SIZE), dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((SIZE, SIZE), dtype=torch.float32).to(torch.bfloat16)

    tiles_m, tiles_n = SIZE // TILE_M, SIZE // TILE_N
    accumulator_bytes = TILE_M * TILE_N * 4
    panels_n = SIZE // PACK_BN
    print(
        f"bf16 {SIZE}x{SIZE} @ {SIZE}x{SIZE} -> f32; "
        f"row-major tiles={TILE_M}x{TILE_N}x{TILE_K}; "
        f"packed panels={panels_n}x{PACK_BN} tiles={PACK_TILE_M}x{PACK_BN}x{PACK_TILE_K}"
    )
    print(
        f"{tiles_m * tiles_n} output tiles over {threads} threads; "
        f"{accumulator_bytes / 1024:.0f} KiB L2-resident accumulator per tile"
    )

    compile_start = time.perf_counter()
    packed_b = helion_block_pack.pack_b(b, PACK_BN)
    helion_packed_result = matmul_bf16_flat_packed(a, packed_b)
    compile_ms = (time.perf_counter() - compile_start) * 1_000
    print(f"Helion first call    {compile_ms:8.3f} ms (includes pack + MLIR JIT)")

    reference = a.to(torch.float32) @ b.to(torch.float32)
    check_numerics("Helion packed (f32)", helion_packed_result, reference)
    row_major_result = matmul_bf16_mlir(a, b)
    check_numerics("Helion row-major", row_major_result, reference)
    # torch.mm keeps the bf16 output dtype, so its error is dominated by the
    # final rounding rather than by the accumulation.
    check_numerics("PyTorch eager (bf16)", torch.mm(a, b), reference)

    with torch.inference_mode():
        pack_ms = benchmark("1 pack B (helion)", lambda: helion_block_pack.pack_b(b, PACK_BN))
        packed_matmul_ms = benchmark(
            "2 matmul (helion)",
            lambda: matmul_bf16_flat_packed(a, packed_b),
        )
        # Comparable with torch.mm: both stages, result already row-major [M, N].
        packed_total_ms = benchmark(
            "packed e2e (1+2)",
            lambda: matmul_bf16_flat_packed(a, helion_block_pack.pack_b(b, PACK_BN)),
        )
        row_major_ms = benchmark("Helion row-major", lambda: matmul_bf16_mlir(a, b))
        eager_ms = benchmark("PyTorch eager", lambda: torch.mm(a, b))

    flops = 2 * SIZE**3
    stage_sum_ms = pack_ms + packed_matmul_ms
    print(f"stage sum            {stage_sum_ms:8.3f} ms")
    print(f"e2e minus stage sum  {packed_total_ms - stage_sum_ms:8.3f} ms (alloc/cold)")
    print(f"matmul only          {flops / (packed_matmul_ms * 1e6):8.1f} GFLOP/s")
    print(f"packed e2e           {flops / (packed_total_ms * 1e6):8.1f} GFLOP/s")
    print(f"Helion row-major     {flops / (row_major_ms * 1e6):8.1f} GFLOP/s")
    print(f"PyTorch eager        {flops / (eager_ms * 1e6):8.1f} GFLOP/s")
    print(f"packed e2e / PyTorch {eager_ms / packed_total_ms:8.3f}x")
    print(f"packed e2e / row-major {row_major_ms / packed_total_ms:8.3f}x")


if __name__ == "__main__":
    main()
