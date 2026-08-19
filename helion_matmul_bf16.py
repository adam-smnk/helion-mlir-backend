"""4K bf16 matmul benchmark for Helion's MLIR backend.

Tuned for a 2-socket Intel Xeon Platinum 8592+ (Emerald Rapids, 64 cores and
256 MiB of private L2 per socket) running one thread per physical core on a
single socket:

    HELION_MLIR_PIPELINE=1 OMP_NUM_THREADS=64 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    LD_PRELOAD=/lib64/libtcmalloc.so:$LD_PRELOAD \
    PYTHONPATH=~/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH \
    uv run python helion_matmul_bf16.py

The Helion kernel only expresses *cache* blocking. Register blocking, VNNI
packing and AMX tile selection are done by the lighthouse pipeline, which
recognises a `linalg.matmul ins(bf16, bf16) outs(f32)` contraction on an
amx_tile target and lowers it to `tdpbf16ps`.
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


SIZE = 4096
TILE_M = int(os.environ.get("HELION_MATMUL_TILE_M", "128"))
TILE_N = int(os.environ.get("HELION_MATMUL_TILE_N", "1024"))
TILE_K = int(os.environ.get("HELION_MATMUL_TILE_K", "32"))
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
    if any(SIZE % tile for tile in (TILE_M, TILE_N, TILE_K)):
        raise ValueError(
            f"Tile sizes must divide {SIZE} for this fixed-shape benchmark"
        )

    threads = int(os.environ.get("OMP_NUM_THREADS", "64"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    a = torch.randn((SIZE, SIZE), dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((SIZE, SIZE), dtype=torch.float32).to(torch.bfloat16)

    tiles_m, tiles_n = SIZE // TILE_M, SIZE // TILE_N
    accumulator_bytes = TILE_M * TILE_N * 4
    print(
        f"bf16 {SIZE}x{SIZE} @ {SIZE}x{SIZE} -> f32; "
        f"Helion cache tiles={TILE_M}x{TILE_N}x{TILE_K}"
    )
    print(
        f"{tiles_m * tiles_n} output tiles over {threads} threads; "
        f"{accumulator_bytes / 1024:.0f} KiB L2-resident accumulator per tile"
    )

    compile_start = time.perf_counter()
    helion_result = matmul_bf16_mlir(a, b)
    compile_ms = (time.perf_counter() - compile_start) * 1_000
    print(f"Helion first call    {compile_ms:8.3f} ms (includes MLIR JIT)")

    reference = a.to(torch.float32) @ b.to(torch.float32)
    check_numerics("Helion MLIR (f32)", helion_result, reference)
    # torch.mm keeps the bf16 output dtype, so its error is dominated by the
    # final rounding rather than by the accumulation.
    check_numerics("PyTorch eager (bf16)", torch.mm(a, b), reference)

    with torch.inference_mode():
        helion_ms = benchmark("Helion MLIR (AMX)", lambda: matmul_bf16_mlir(a, b))
        eager_ms = benchmark("PyTorch eager", lambda: torch.mm(a, b))

    flops = 2 * SIZE**3
    print(f"Helion MLIR (AMX)    {flops / (helion_ms * 1e6):8.1f} GFLOP/s")
    print(f"PyTorch eager        {flops / (eager_ms * 1e6):8.1f} GFLOP/s")
    print(f"Helion / PyTorch     {eager_ms / helion_ms:8.3f}x")


if __name__ == "__main__":
    main()
