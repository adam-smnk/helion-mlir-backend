"""4K bf16 matmul benchmark for Helion's MLIR backend.

Tuned for a 2-socket Intel Xeon Platinum 8592+ (Emerald Rapids, 64 cores and
256 MiB of private L2 per socket) running one thread per physical core on a
single socket:

    HELION_MLIR_PIPELINE=1 OMP_NUM_THREADS=64 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    LD_PRELOAD=/lib64/libtcmalloc.so:$LD_PRELOAD \
    PYTHONPATH=~/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH \
    uv run python helion_matmul_bf16.py

The optimized packed path uses a true blocked 4D contraction:
``[MB, KB, BM, BK] @ [NB, KB, BK, BN] -> [MB, NB, BM, BN]``.
That is the shape lighthouse's block-packing pipeline expects: the major block
axes and minor register-tile axes are all part of a single `linalg.contract`,
so AMX lowering can see the complete M/N/K geometry at once.

The older host-panel path remains as a fallback/comparison point. It computes
one packed-B panel at a time with a plain 2D contract and writes directly into
a flat `[M, N]` result, avoiding problematic in-kernel panel stores at the cost
of one kernel launch per panel.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import helion_block_pack
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
# Panel width for the packed path: each panel is one host-side matmul call, so
# this also sets the per-panel N tile. 256 was the sweep optimum (128/512/1024
# were all slower) at 4096x4096, 4 threads.
PACK_BN = int(os.environ.get("HELION_MATMUL_PACK_BN", "256"))
PACK_TILE_M = int(os.environ.get("HELION_MATMUL_PACK_TILE_M", "128"))
PACK_TILE_K = int(os.environ.get("HELION_MATMUL_PACK_TILE_K", "32"))
MMT4D_BLOCK_M = int(os.environ.get("HELION_MATMUL_MMT4D_BLOCK_M", "32"))
MMT4D_BLOCK_N = int(os.environ.get("HELION_MATMUL_MMT4D_BLOCK_N", "32"))
MMT4D_BLOCK_K = int(os.environ.get("HELION_MATMUL_MMT4D_BLOCK_K", "32"))
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

    The result stays f32 here because the wide row-major bf16 epilogue still
    fails lowering; the packed mmt4d path below provides torch-like bf16 output.
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
            acc = acc + torch.einsum(
                "mk,kn->mn", a[tile_m, tile_k], b_panel[tile_k, tile_n]
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def matmul_bf16_mmt4d_mlir(a4: Tensor, b4: Tensor) -> Tensor:
    """Blocked 4D bf16 matmul, matching lighthouse's packed contract shape."""
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"

    out = torch.empty(
        (blocks_m, blocks_n, block_m, block_n),
        dtype=a4.dtype,
        device=a4.device,
    )
    for tile_blocks_m, tile_blocks_n in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros(
            [tile_blocks_m, tile_blocks_n, block_m, block_n], dtype=torch.float32
        )
        acc = acc + torch.einsum(
            "akmc,bkcn->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        out[tile_blocks_m, tile_blocks_n, :, :] = acc.to(a4.dtype)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def matmul_bf16_mmt4d_merged_unpack_mlir(a4: Tensor, b4: Tensor) -> Tensor:
    """Blocked 4D bf16 matmul that stores in row-major-viewable block order."""
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"

    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n),
        dtype=a4.dtype,
        device=a4.device,
    )
    for tile_blocks_m, tile_blocks_n in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros(
            [tile_blocks_m, tile_blocks_n, block_m, block_n], dtype=torch.float32
        )
        acc = acc + torch.einsum(
            "akmc,bkcn->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        out[tile_blocks_m, :, tile_blocks_n, :] = acc.permute(0, 2, 1, 3).to(a4.dtype)
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

    out = torch.empty((m, panels_n * bn), dtype=a.dtype, device=a.device)
    for panel in range(panels_n):
        out[:, panel * bn : (panel + 1) * bn] = matmul_bf16_panel_mlir(
            a, packed_b[panel]
        )
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
)
def pack_a_mmt4d_kernel(a4_src: Tensor) -> Tensor:
    """Pack A from ``[M/BM, BM, K/BK, BK]`` to ``[M/BM, K/BK, BM, BK]``."""
    blocks_m, block_m, blocks_k, block_k = a4_src.shape
    out = torch.empty(
        (blocks_m, blocks_k, block_m, block_k),
        dtype=a4_src.dtype,
        device=a4_src.device,
    )
    for block_mi, block_ki, tile_m, tile_k in hl.tile(
        [blocks_m, blocks_k, block_m, block_k]
    ):
        out[block_mi, block_ki, tile_m, tile_k] = a4_src[
            block_mi, tile_m, block_ki, tile_k
        ].permute(0, 2, 1, 3)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
)
def unpack_mmt4d_kernel(out4: Tensor) -> Tensor:
    """Unpack ``[M/BM, N/BN, BM, BN]`` to ``[M/BM, BM, N/BN, BN]``."""
    blocks_m, blocks_n, block_m, block_n = out4.shape
    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n),
        dtype=out4.dtype,
        device=out4.device,
    )
    for block_mi, block_ni, tile_m, tile_n in hl.tile(
        [blocks_m, blocks_n, block_m, block_n]
    ):
        out[block_mi, tile_m, block_ni, tile_n] = out4[
            block_mi, block_ni, tile_m, tile_n
        ].permute(0, 2, 1, 3)
    return out


def pack_a_mmt4d(a: Tensor) -> Tensor:
    """Pack row-major A into ``[M/BM, K/BK, BM, BK]``."""
    m, k = a.shape
    return pack_a_mmt4d_kernel(
        a.view(
            m // MMT4D_BLOCK_M,
            MMT4D_BLOCK_M,
            k // MMT4D_BLOCK_K,
            MMT4D_BLOCK_K,
        ).contiguous()
    )


def pack_b_mmt4d(b: Tensor) -> Tensor:
    """Pack row-major B into ``[N/BN, K/BK, BK, BN]``."""
    packed_panel = helion_block_pack.pack_b(b, MMT4D_BLOCK_N)
    k, n = b.shape
    return packed_panel.view(
        n // MMT4D_BLOCK_N, k // MMT4D_BLOCK_K, MMT4D_BLOCK_K, MMT4D_BLOCK_N
    )


def unpack_mmt4d(out4: Tensor) -> Tensor:
    """Return row-major ``[M, N]`` from ``[M/BM, N/BN, BM, BN]``."""
    blocks_m, blocks_n, block_m, block_n = out4.shape
    return (
        unpack_mmt4d_kernel(out4)
        .contiguous()
        .view(blocks_m * block_m, blocks_n * block_n)
    )


def view_merged_mmt4d(out4: Tensor) -> Tensor:
    """Return row-major ``[M, N]`` from ``[M/BM, BM, N/BN, BN]``."""
    blocks_m, block_m, blocks_n, block_n = out4.shape
    return out4.view(blocks_m * block_m, blocks_n * block_n)


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


def check_bf16_semantics(name: str, actual: Tensor, reference_bf16: Tensor) -> None:
    """Check the result dtype and rounded bf16 values against ``torch.mm``."""
    exact_mismatches = (actual != reference_bf16).sum().item()
    abs_err = (actual.float() - reference_bf16.float()).abs()
    print(
        f"{name:20s} dtype {actual.dtype}, exact mismatches {exact_mismatches}, "
        f"max bf16 delta {abs_err.max().item():.3e}"
    )
    assert actual.dtype == reference_bf16.dtype
    torch.testing.assert_close(actual, reference_bf16, rtol=3e-2, atol=1.0)


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
    if any(
        block != AMX_PARALLEL_TILE
        for block in (MMT4D_BLOCK_M, MMT4D_BLOCK_N, MMT4D_BLOCK_K)
    ):
        raise ValueError(
            f"MMT4D_BLOCK_M/N/K must all be {AMX_PARALLEL_TILE} for AMX bf16"
        )
    if any(SIZE % tile for tile in (TILE_M, TILE_N, TILE_K)):
        raise ValueError(
            f"Tile sizes must divide {SIZE} for this fixed-shape benchmark"
        )
    if any(
        SIZE % tile
        for tile in (
            PACK_TILE_M,
            PACK_BN,
            PACK_TILE_K,
            MMT4D_BLOCK_M,
            MMT4D_BLOCK_N,
            MMT4D_BLOCK_K,
        )
    ):
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
        f"bf16 {SIZE}x{SIZE} @ {SIZE}x{SIZE}; f32 accum, bf16 packed outputs; "
        f"row-major tiles={TILE_M}x{TILE_N}x{TILE_K}; "
        f"host panels={panels_n}x{PACK_BN}; "
        f"mmt4d blocks={MMT4D_BLOCK_M}x{MMT4D_BLOCK_N}x{MMT4D_BLOCK_K}"
    )
    print(
        f"{tiles_m * tiles_n} output tiles over {threads} threads; "
        f"{accumulator_bytes / 1024:.0f} KiB L2-resident accumulator per tile"
    )

    compile_start = time.perf_counter()
    a4 = pack_a_mmt4d(a)
    b4 = pack_b_mmt4d(b)
    mmt4d_blocked_result = matmul_bf16_mmt4d_mlir(a4, b4)
    mmt4d_merged_result = matmul_bf16_mmt4d_merged_unpack_mlir(a4, b4)
    packed_b = helion_block_pack.pack_b(b, PACK_BN)
    helion_panel_result = matmul_bf16_flat_packed(a, packed_b)
    compile_ms = (time.perf_counter() - compile_start) * 1_000
    print(f"Helion first call    {compile_ms:8.3f} ms (includes packs + MLIR JIT)")

    reference = a.to(torch.float32) @ b.to(torch.float32)
    torch_result = torch.mm(a, b)
    helion_mmt4d_result = unpack_mmt4d(mmt4d_blocked_result)
    helion_merged_result = view_merged_mmt4d(mmt4d_merged_result)
    check_numerics("Helion mmt4d (bf16)", helion_mmt4d_result, reference)
    check_numerics("Helion merged (bf16)", helion_merged_result, reference)
    check_numerics("Helion panel (bf16)", helion_panel_result, reference)
    check_bf16_semantics("Helion mmt4d", helion_mmt4d_result, torch_result)
    check_bf16_semantics("Helion merged", helion_merged_result, torch_result)
    check_bf16_semantics("Helion panel", helion_panel_result, torch_result)
    row_major_result = matmul_bf16_mlir(a, b)
    check_numerics("Helion row-major", row_major_result, reference)
    # torch.mm keeps the bf16 output dtype, so its error is dominated by the
    # final rounding rather than by the accumulation.
    check_numerics("PyTorch eager (bf16)", torch_result, reference)

    with torch.inference_mode():
        pack_a_ms = benchmark("1a pack A (helion)", lambda: pack_a_mmt4d(a))
        pack_b_mmt4d_ms = benchmark("1b pack B (helion)", lambda: pack_b_mmt4d(b))
        mmt4d_matmul_ms = benchmark(
            "2a mmt4d (helion)",
            lambda: matmul_bf16_mmt4d_mlir(a4, b4),
        )
        mmt4d_merged_ms = benchmark(
            "2b merged (helion)",
            lambda: matmul_bf16_mmt4d_merged_unpack_mlir(a4, b4),
        )
        mmt4d_unpack_ms = benchmark(
            "3a unpack (helion)",
            lambda: unpack_mmt4d(mmt4d_blocked_result),
        )
        mmt4d_total_ms = benchmark(
            "mmt4d e2e (1+2+3)",
            lambda: unpack_mmt4d(
                matmul_bf16_mmt4d_mlir(pack_a_mmt4d(a), pack_b_mmt4d(b))
            ),
        )
        mmt4d_merged_total_ms = benchmark(
            "merged e2e (1+2)",
            lambda: view_merged_mmt4d(
                matmul_bf16_mmt4d_merged_unpack_mlir(pack_a_mmt4d(a), pack_b_mmt4d(b))
            ),
        )
        pack_ms = benchmark(
            "1 pack B (helion)", lambda: helion_block_pack.pack_b(b, PACK_BN)
        )
        panel_matmul_ms = benchmark(
            "2 panel (helion)",
            lambda: matmul_bf16_flat_packed(a, packed_b),
        )
        # Comparable with torch.mm: both stages, result already row-major [M, N].
        panel_total_ms = benchmark(
            "panel e2e (1+2)",
            lambda: matmul_bf16_flat_packed(a, helion_block_pack.pack_b(b, PACK_BN)),
        )
        row_major_ms = benchmark("Helion row-major", lambda: matmul_bf16_mlir(a, b))
        eager_ms = benchmark("PyTorch eager", lambda: torch.mm(a, b))

    flops = 2 * SIZE**3
    mmt4d_stage_sum_ms = pack_a_ms + pack_b_mmt4d_ms + mmt4d_matmul_ms + mmt4d_unpack_ms
    merged_stage_sum_ms = pack_a_ms + pack_b_mmt4d_ms + mmt4d_merged_ms
    panel_stage_sum_ms = pack_ms + panel_matmul_ms
    print(f"mmt4d stage sum      {mmt4d_stage_sum_ms:8.3f} ms")
    print(
        f"mmt4d e2e - stages   {mmt4d_total_ms - mmt4d_stage_sum_ms:8.3f} ms (alloc/cold)"
    )
    print(f"merged stage sum     {merged_stage_sum_ms:8.3f} ms")
    print(
        f"merged e2e - stages  {mmt4d_merged_total_ms - merged_stage_sum_ms:8.3f} ms (alloc/cold)"
    )
    print(f"panel stage sum      {panel_stage_sum_ms:8.3f} ms")
    print(
        f"panel e2e - stages   {panel_total_ms - panel_stage_sum_ms:8.3f} ms (alloc/cold)"
    )
    print(f"mmt4d matmul only    {flops / (mmt4d_matmul_ms * 1e6):8.1f} GFLOP/s")
    print(f"mmt4d e2e            {flops / (mmt4d_total_ms * 1e6):8.1f} GFLOP/s")
    print(f"merged matmul only   {flops / (mmt4d_merged_ms * 1e6):8.1f} GFLOP/s")
    print(f"merged e2e           {flops / (mmt4d_merged_total_ms * 1e6):8.1f} GFLOP/s")
    print(f"panel matmul only    {flops / (panel_matmul_ms * 1e6):8.1f} GFLOP/s")
    print(f"panel e2e            {flops / (panel_total_ms * 1e6):8.1f} GFLOP/s")
    print(f"Helion row-major     {flops / (row_major_ms * 1e6):8.1f} GFLOP/s")
    print(f"PyTorch eager        {flops / (eager_ms * 1e6):8.1f} GFLOP/s")
    print(f"mmt4d e2e / PyTorch  {eager_ms / mmt4d_total_ms:8.3f}x")
    print(f"merged e2e / PyTorch {eager_ms / mmt4d_merged_total_ms:8.3f}x")
    print(f"mmt4d e2e / row-major {row_major_ms / mmt4d_total_ms:8.3f}x")


if __name__ == "__main__":
    main()
