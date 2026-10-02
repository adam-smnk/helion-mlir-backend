"""
GEMM with Fused Packing on the MLIR Backend

BLAS libraries (GotoBLAS, BLIS, oneDNN) do not multiply row-major operands
directly: they first pack panels of A and B into a blocked layout, so the inner
kernel streams contiguous blocks that fit in registers and caches. This example
writes that structure as one Helion kernel of two phases:

1. Packing: A ``[M, K] -> [M/32, K/32, 32, 32]`` and B ``[K, N] ->
   [N/32, K/32, 32, 32]``, one 32x32 block per loop iteration. Loads past the
   end of an operand read zeros, so padding to whole blocks is part of the pass.
2. ``hl.barrier()``, then the blocked (mmt4d-style) contraction: each tile of
   ``BM x BN`` output blocks accumulates ``BK`` packed K blocks per step and
   stores into ``[M/32, 32, N/32, 32]``, which is row-major ``[M, N]`` padded to
   whole blocks: the result is a view of it.

The blocked layout is the one AMX's bf16 tiles read (see ``vnni_packing_mlir.py``
for the VNNI layout of B). Here everything is f32.

``gemm`` picks tile sizes, in blocks, that divide the block counts: a partial
tile of blocks takes the masked path, which is slower. The checks compare against
``torch.matmul`` on aligned, padded and odd shapes, and the timings are a quick
ballpark against eager PyTorch and a plain Helion matmul on row-major tiles. Run
with ``LD_PRELOAD=libtcmalloc.so.4`` to keep large outputs from being
page-faulted in on every call.
"""

from __future__ import annotations

import os

# Before torch and the MLIR runtime start their OpenMP thread pools.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))

from itertools import starmap
import statistics
import time
from typing import Callable

import helion
import helion.language as hl
import torch

import helion_mlir_backend  # noqa: F401

# Tile sizes of the contraction, in 32x32 blocks: rows, columns, K step.
TILE_BLOCKS = (2, 4, 2)


def _gemm_packed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` with both operands packed into 32x32 blocks in the same kernel."""
    m, k = a.shape
    _, n = b.shape
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


_KERNELS: dict[tuple[int, int, int], helion.Kernel] = {}


def _divisor_at_most(extent: int, limit: int) -> int:
    return max(d for d in range(1, min(extent, limit) + 1) if extent % d == 0)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for 2-D f32 operands, with tiles that divide the block counts."""
    (m, k), n = a.shape, b.shape[1]
    counts = ((m + 31) // 32, (n + 31) // 32, (k + 31) // 32)
    blocks = tuple(starmap(_divisor_at_most, zip(counts, TILE_BLOCKS, strict=True)))
    if blocks not in _KERNELS:
        _KERNELS[blocks] = helion.kernel(
            _gemm_packed,
            backend="mlir",
            static_shapes=True,
            config=helion.Config(block_sizes=list(blocks)),
        )
    return _KERNELS[blocks](a, b)


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[128, 512, 64]),
)
def gemm_plain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` on row-major tiles, for comparison."""
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc
    return out


def measure(fn: Callable[..., torch.Tensor], *args: torch.Tensor) -> float:
    """Milliseconds per call: the median of 5 rounds of 3 calls, after warm-up."""
    fn(*args)
    rounds = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(3):
            fn(*args)
        rounds.append((time.perf_counter() - start) / 3)
    return statistics.median(rounds) * 1e3


def check(shapes: list[tuple[int, int, int]]) -> None:
    for m, k, n in shapes:
        a, b = torch.randn(m, k), torch.randn(k, n)
        expected = a @ b
        wrong = [
            name
            for name, fn in [("gemm", gemm), ("gemm_plain", gemm_plain)]
            if not torch.allclose(fn(a, b), expected, atol=1e-3, rtol=1e-3)
        ]
        print(f"  {m}x{k}x{n}: {'ok' if not wrong else 'WRONG: ' + ', '.join(wrong)}")


def benchmark(shapes: list[tuple[int, int, int]]) -> None:
    approaches = [("torch", torch.matmul), ("gemm", gemm), ("gemm_plain", gemm_plain)]
    print(f"  {'M x K x N':>16}" + "".join(f"{name:>20}" for name, _ in approaches))
    for m, k, n in shapes:
        a, b = torch.randn(m, k), torch.randn(k, n)
        cells = []
        for _, fn in approaches:
            ms = measure(fn, a, b)
            cells.append(f"{ms:7.2f} ms {2 * m * n * k / ms / 1e6:5.0f} GF/s")
        shape = f"{m}x{k}x{n}"
        print(f"  {shape:>16}" + "".join(f"{c:>20}" for c in cells))


def main() -> None:
    torch.manual_seed(0)
    print("Correctness against torch.matmul (f32):")
    check([(64, 96, 128), (70, 45, 50), (1, 33, 65), (200, 300, 100)])
    print(f"\nf32 GEMM, OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}:")
    benchmark([(2048, 2048, 2048), (1024, 3072, 1536), (1000, 3000, 1500)])


if __name__ == "__main__":
    main()
