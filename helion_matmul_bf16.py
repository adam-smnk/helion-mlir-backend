"""4K bf16 matmul benchmark for Helion's MLIR backend.

Tuned for a 2-socket Intel Xeon Platinum 8592+ (Emerald Rapids, 64 cores and
256 MiB of private L2 per socket) running one thread per physical core on a
single socket:

    HELION_MLIR_PIPELINE=1 OMP_NUM_THREADS=64 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    LD_PRELOAD=/lib64/libtcmalloc.so:$LD_PRELOAD \
    PYTHONPATH=~/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH \
    uv run python helion_matmul_bf16.py

The optimized path packs the RHS into column panels before matmul. Register
blocking, VNNI packing and AMX tile selection are done by the lighthouse
pipeline, which recognises a `linalg.matmul ins(bf16, bf16) outs(f32)`
contraction on an amx_tile target and lowers it to `tdpbf16ps`.
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

import helion_mlir_backend  # noqa: F401

# Helion probes tile shapes during tracing and torch logs the recovered failures.
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)

if TYPE_CHECKING:
    from collections.abc import Callable


SIZE = 4096
TILE_M = int(os.environ.get("HELION_MATMUL_TILE_M", "128"))
TILE_N = int(os.environ.get("HELION_MATMUL_TILE_N", "1024"))
TILE_K = int(os.environ.get("HELION_MATMUL_TILE_K", "32"))
PACK_TILE_M = int(os.environ.get("HELION_MATMUL_PACK_TILE_M", "16"))
PACK_TILE_N = int(os.environ.get("HELION_MATMUL_PACK_TILE_N", "32"))
PACK_TILE_K = int(os.environ.get("HELION_MATMUL_PACK_TILE_K", "32"))
# Cache-level M block; the packed grid iterates (m_block, n_panel) so an A block
# stays resident while all N panels stream past it.
PACK_BLOCK_M = int(os.environ.get("HELION_MATMUL_PACK_BLOCK_M", "512"))
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
    config=helion.Config(block_sizes=[1, PACK_TILE_K, PACK_TILE_N]),
)
def pack_b_panels_bf16(b3: Tensor) -> Tensor:
    """Pack ``B`` from ``[K, N/BN, BN]`` to ``[N/BN, K, BN]``."""
    k, panels_n, block_n = b3.shape
    out = torch.empty((panels_n, k, block_n), dtype=b3.dtype, device=b3.device)
    for panel_n in hl.grid(panels_n):
        for tile_k in hl.tile(k):
            out[panel_n, tile_k, :] = b3[tile_k, panel_n, :]
    return out


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
    config=helion.Config(block_sizes=[PACK_TILE_M, PACK_TILE_K]),
)
def matmul_bf16_packed_b_mlir(a3: Tensor, packed_b: Tensor) -> Tensor:
    """Blocked ``A @ B`` from ``[M/BM, BM, K]`` and packed RHS ``[N/BN, K, BN]``.

    `PACK_TILE_M`/`PACK_TILE_N` are pinned to 16x32: larger register tiles either
    fail AMX conversion or abort in MLIR, so the inner contraction is exactly two
    `tdpbf16ps` accumulator chains held in AMX tile registers across the K loop.
    """
    m_blocks, block_m, k = a3.shape
    panels_n, k2, block_n = packed_b.shape
    assert k == k2, "matmul dimension mismatch"

    out = torch.empty(
        (m_blocks, panels_n, block_m, block_n),
        dtype=torch.float32,
        device=a3.device,
    )
    for m_block, panel_n in hl.grid([m_blocks, panels_n]):
        for tile_m in hl.tile(block_m):
            acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(
                    acc,
                    a3[m_block, tile_m, tile_k],
                    packed_b[panel_n, tile_k, :],
                )
            out[m_block, panel_n, tile_m, :] = acc
    return out


def pack_a_host_shape(a: Tensor) -> Tensor:
    """View ``A`` as ``[M/BM, BM, K]``; row-major A makes this a free reshape."""
    m, k = a.shape
    return a.view(m // PACK_BLOCK_M, PACK_BLOCK_M, k)


def pack_b_host_shape(b: Tensor) -> Tensor:
    """View ``B`` as ``[K, N/BN, BN]`` before device-side panel packing."""
    k, n = b.shape
    return b.view(k, n // PACK_TILE_N, PACK_TILE_N).contiguous()


def unpack_blocked(blocked_out: Tensor) -> Tensor:
    """Convert ``[M/BM, N/BN, BM, BN]`` back to row-major ``[M, N]``.

    Eager torch, not a Helion kernel: the required store pattern indexes a rank-4
    destination as (scalar, tile, scalar, full), which the MLIR backend either
    miscompiles or crashes on.
    """
    m_blocks, panels_n, block_m, block_n = blocked_out.shape
    return (
        blocked_out.permute(0, 2, 1, 3)
        .contiguous()
        .view(m_blocks * block_m, panels_n * block_n)
    )


def matmul_bf16_packed_end_to_end(a3: Tensor, b3: Tensor) -> Tensor:
    """Full packed path returning row-major ``[M, N]``, as ``torch.mm`` does."""
    return unpack_blocked(matmul_bf16_packed_b_mlir(a3, pack_b_panels_bf16(b3)))


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
    if PACK_TILE_N != AMX_PARALLEL_TILE:
        raise ValueError(f"PACK_TILE_N must be {AMX_PARALLEL_TILE} for AMX bf16")
    if SIZE % PACK_BLOCK_M or PACK_BLOCK_M % PACK_TILE_M:
        raise ValueError(
            f"PACK_BLOCK_M must divide {SIZE} and be a multiple of PACK_TILE_M"
        )
    if any(SIZE % tile for tile in (TILE_M, TILE_N, TILE_K)):
        raise ValueError(
            f"Tile sizes must divide {SIZE} for this fixed-shape benchmark"
        )
    if any(SIZE % tile for tile in (PACK_TILE_M, PACK_TILE_N, PACK_TILE_K)):
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
    print(
        f"bf16 {SIZE}x{SIZE} @ {SIZE}x{SIZE} -> f32; "
        f"row-major tiles={TILE_M}x{TILE_N}x{TILE_K}; "
        f"packed block_m={PACK_BLOCK_M} "
        f"tiles={PACK_TILE_M}x{PACK_TILE_N}x{PACK_TILE_K}"
    )
    print(
        f"{tiles_m * tiles_n} output tiles over {threads} threads; "
        f"{accumulator_bytes / 1024:.0f} KiB L2-resident accumulator per tile"
    )

    b3 = pack_b_host_shape(b)
    a3 = pack_a_host_shape(a)
    compile_start = time.perf_counter()
    packed_b = pack_b_panels_bf16(b3)
    helion_blocked_result = matmul_bf16_packed_b_mlir(a3, packed_b)
    compile_ms = (time.perf_counter() - compile_start) * 1_000
    print(f"Helion first call    {compile_ms:8.3f} ms (includes pack + MLIR JIT)")

    reference = a.to(torch.float32) @ b.to(torch.float32)
    helion_result = unpack_blocked(helion_blocked_result)
    check_numerics("Helion packed (f32)", helion_result, reference)
    row_major_result = matmul_bf16_mlir(a, b)
    check_numerics("Helion row-major", row_major_result, reference)
    # torch.mm keeps the bf16 output dtype, so its error is dominated by the
    # final rounding rather than by the accumulation.
    check_numerics("PyTorch eager (bf16)", torch.mm(a, b), reference)

    with torch.inference_mode():
        pack_ms = benchmark("1 pack B (helion)", lambda: pack_b_panels_bf16(b3))
        packed_matmul_ms = benchmark(
            "2 matmul (helion)",
            lambda: matmul_bf16_packed_b_mlir(a3, packed_b),
        )
        unpack_ms = benchmark(
            "3 unpack (eager)",
            lambda: unpack_blocked(helion_blocked_result),
        )
        # Comparable with torch.mm: all three stages, result row-major [M, N].
        packed_total_ms = benchmark(
            "packed e2e (1+2+3)",
            lambda: matmul_bf16_packed_end_to_end(a3, b3),
        )
        row_major_ms = benchmark("Helion row-major", lambda: matmul_bf16_mlir(a, b))
        eager_ms = benchmark("PyTorch eager", lambda: torch.mm(a, b))

    flops = 2 * SIZE**3
    stage_sum_ms = pack_ms + packed_matmul_ms + unpack_ms
    print(f"stage sum            {stage_sum_ms:8.3f} ms")
    print(f"e2e minus stage sum  {packed_total_ms - stage_sum_ms:8.3f} ms (alloc/cold)")
    print(f"matmul only          {flops / (packed_matmul_ms * 1e6):8.1f} GFLOP/s")
    print(f"packed e2e           {flops / (packed_total_ms * 1e6):8.1f} GFLOP/s")
    print(f"Helion row-major     {flops / (row_major_ms * 1e6):8.1f} GFLOP/s")
    print(f"PyTorch eager        {flops / (eager_ms * 1e6):8.1f} GFLOP/s")
    print(f"packed e2e / PyTorch {eager_ms / packed_total_ms:8.3f}x")


if __name__ == "__main__":
    main()
