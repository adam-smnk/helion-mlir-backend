"""1K f32 matmul benchmark for Helion's MLIR backend.

The default 128x512x64 cache tiles were optimized on an 11th Gen Intel Core
i7-11850H (8 cores, AVX-512, 1.25 MiB private L2 per core), using four OpenMP
threads, taskset or KMP_AFFINITY setting, and tcmalloc via LD_PRELOAD.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Callable


SIZE = 1024
TILE_M = int(os.environ.get("HELION_MATMUL_TILE_M", "128"))
TILE_N = int(os.environ.get("HELION_MATMUL_TILE_N", "512"))
TILE_K = int(os.environ.get("HELION_MATMUL_TILE_K", "64"))
WARMUP_ITERS = 5
BENCHMARK_ITERS = 20
SAMPLES = 7


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[TILE_M, TILE_N, TILE_K]),
)
def matmul_mlir(x: Tensor, y: Tensor) -> Tensor:
    """Compute a f32 matrix product using L2-resident output tiles.

    Each 128x512x64 update touches 416 KiB across the A, B, and C tiles,
    comfortably below the 1.25 MiB private L2 cache on the target CPU. The
    outer tiles are independent, while the K loop is explicitly sequential so
    the MLIR pipeline sees an accumulate-form linalg contraction.
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
    print(f"{name:16s} {median_ms:8.3f} ms")
    return median_ms


def main() -> None:
    if os.environ.get("HELION_MLIR_PIPELINE") != "1":
        raise RuntimeError("Set HELION_MLIR_PIPELINE=1 to use the vectorizing pipeline")
    if len({TILE_M, TILE_N, TILE_K}) != 3:
        raise ValueError(
            "MLIR tile sizes must be distinct to preserve loop-index mapping"
        )
    if any(SIZE % tile != 0 for tile in (TILE_M, TILE_N, TILE_K)):
        raise ValueError(
            f"Tile sizes must divide {SIZE} for this fixed-shape benchmark"
        )

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(0)
    a = torch.randn((SIZE, SIZE), dtype=torch.float32)
    b = torch.randn((SIZE, SIZE), dtype=torch.float32)

    print(
        f"f32 {SIZE}x{SIZE} @ {SIZE}x{SIZE}; "
        f"Helion tiles={TILE_M}x{TILE_N}x{TILE_K}; "
        f"PyTorch threads={torch.get_num_threads()}"
    )

    compile_start = time.perf_counter()
    helion_result = matmul_mlir(a, b)
    compile_ms = (time.perf_counter() - compile_start) * 1_000
    reference = torch.mm(a, b)
    torch.testing.assert_close(helion_result, reference, rtol=1e-4, atol=1e-4)
    print(f"Helion first call   {compile_ms:8.3f} ms (includes MLIR JIT)")

    with torch.inference_mode():
        helion_ms = benchmark("Helion MLIR", lambda: matmul_mlir(a, b))
        eager_ms = benchmark("PyTorch eager", lambda: torch.mm(a, b))

    flops = 2 * SIZE**3
    helion_gflops = flops / (helion_ms * 1e6)
    eager_gflops = flops / (eager_ms * 1e6)
    print(f"Helion MLIR         {helion_gflops:8.1f} GFLOP/s")
    print(f"PyTorch eager       {eager_gflops:8.1f} GFLOP/s")
    print(f"Helion / PyTorch    {eager_ms / helion_ms:8.3f}x")


if __name__ == "__main__":
    main()
