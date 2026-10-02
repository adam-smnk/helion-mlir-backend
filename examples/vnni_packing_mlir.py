"""
VNNI Block Packing for AMX Matmuls with the MLIR Backend

AMX's bf16 matmul (``tdpbf16ps``) reads its B tile in VNNI layout: each 32-bit
row element holds a pair of bf16 values adjacent in K. A blocked matmul packs B
into 32x32 blocks anyway (see ``block_packing_mlir.py``), so the same pass can
store each block in VNNI layout:

    B [K, N] -> [N/32, K/32, 16, 32, 2]   out[j, i, r, n, p] = B[32i + 2r + p, 32j + n]

A block (j, i) is then the ``[16, 64]`` bf16 AMX tile of B[32i:32i+32,
32j:32j+32]. Extents that are not multiples of 32 are zero-padded. A needs no
VNNI step: its K pairs are already adjacent in a row-major block.

``pack_b_vnni`` (B given as ``[K, N]``) and ``pack_b_t_vnni`` (B given as
``[N, K]``, e.g. ``nn.Linear`` weights) do it in one Helion kernel: each
iteration of the loop over the packed blocks loads one 32x32 block, padded with
zeros past the operand's end, splits K into pairs with a reshape and moves the
pair dimension innermost with a permute.

The benchmark compares them with eager PyTorch (one pad and one copy) and with
plain block packing without VNNI, which moves the same bytes. It uses f32: the
layout logic is the same, and on CPUs without AVX512_BF16 LLVM scalarizes masked
bf16 loads, which would dominate the bf16 timings of padded operands. The
correctness check covers bf16 too.

The benchmark pins OpenMP threads (without that, timings vary several-fold
between runs), and every approach allocates its output on each call: run with
``LD_PRELOAD=libtcmalloc.so.4`` to keep large outputs from being page-faulted in
on every call.
"""

from __future__ import annotations

import os

# Before torch and the MLIR runtime start their OpenMP thread pools.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))

import statistics
import time
from typing import Callable

import helion
import helion.language as hl
import torch

import helion_mlir_backend  # noqa: F401  (registers the "mlir" backend)

BLOCK = 32
VNNI = 2
CONFIG = helion.Config(block_sizes=[], mlir_pipeline="opt")


# ---------------------------------------------------------------------------
# The kernels: blocking, padding and VNNI interleaving in one pass. Block sizes
# are literals: Helion cannot resolve a block size from a global name.
# ---------------------------------------------------------------------------


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_b_vnni(b: torch.Tensor) -> torch.Tensor:
    """``[K, N] -> [N/32, K/32, 16, 32, 2]``, zero-padded."""
    k, n = b.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b[tk, tn].reshape(tk.block_size // 2, 2, tn.block_size)
        out[tn.id, tk.id, :, :, :] = pairs.permute(0, 2, 1)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_b_t_vnni(b_t: torch.Tensor) -> torch.Tensor:
    """B given as ``[N, K]``: the same layout as ``pack_b_vnni(b_t.T)``."""
    n, k = b_t.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b_t.dtype, device=b_t.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b_t[tn, tk].reshape(tn.block_size, tk.block_size // 2, 2)
        out[tn.id, tk.id, :, :, :] = pairs.permute(1, 0, 2)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_b(b: torch.Tensor) -> torch.Tensor:
    """Plain block packing, ``[K, N] -> [N/32, K/32, 32, 32]``: the same bytes moved."""
    k, n = b.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b[tk, tn]
    return out


# ---------------------------------------------------------------------------
# Eager PyTorch references.
# ---------------------------------------------------------------------------


def _pad(x: torch.Tensor) -> torch.Tensor:
    rows, cols = (-x.shape[0]) % BLOCK, (-x.shape[1]) % BLOCK
    return torch.nn.functional.pad(x, (0, cols, 0, rows))


def pack_b_vnni_eager(b: torch.Tensor) -> torch.Tensor:
    padded = _pad(b)
    kb, nb = padded.shape[0] // BLOCK, padded.shape[1] // BLOCK
    blocks = padded.reshape(kb, BLOCK // VNNI, VNNI, nb, BLOCK)
    return blocks.permute(3, 0, 1, 4, 2).contiguous()


def pack_b_t_vnni_eager(b_t: torch.Tensor) -> torch.Tensor:
    padded = _pad(b_t)
    nb, kb = padded.shape[0] // BLOCK, padded.shape[1] // BLOCK
    blocks = padded.reshape(nb, BLOCK, kb, BLOCK // VNNI, VNNI)
    return blocks.permute(0, 2, 3, 1, 4).contiguous()


def pack_b_eager(b: torch.Tensor) -> torch.Tensor:
    padded = _pad(b)
    kb, nb = padded.shape[0] // BLOCK, padded.shape[1] // BLOCK
    return padded.reshape(kb, BLOCK, nb, BLOCK).permute(2, 0, 1, 3).contiguous()


def vnni_reference(b: torch.Tensor) -> torch.Tensor:
    """The layout by its definition, element by element (small shapes only)."""
    k, n = b.shape
    kb, nb = (k + BLOCK - 1) // BLOCK, (n + BLOCK - 1) // BLOCK
    out = torch.zeros((nb, kb, BLOCK // VNNI, BLOCK, VNNI), dtype=b.dtype)
    for j, i, r, col, p in torch.cartesian_prod(
        *(torch.arange(size) for size in out.shape)
    ).tolist():
        row = BLOCK * i + VNNI * r + p
        if row < k and BLOCK * j + col < n:
            out[j, i, r, col, p] = b[row, BLOCK * j + col]
    return out


# ---------------------------------------------------------------------------
# Checks and timings.
# ---------------------------------------------------------------------------


def measure(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> float:
    """Milliseconds per call: the median of 10 rounds of the median of 10 calls."""
    for _ in range(3):
        fn(x)
    rounds = []
    for _ in range(10):
        times = []
        for _ in range(10):
            start = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - start)
        rounds.append(statistics.median(times))
    return statistics.median(rounds) * 1e3


def check(dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int]]) -> None:
    for dtype in dtypes:
        for rows, cols in shapes:
            x = torch.randn(rows, cols).to(dtype)
            expected = vnni_reference(x)
            got = {
                "eager": pack_b_vnni_eager(x),
                "eager transposed": pack_b_t_vnni_eager(x.T.contiguous()),
                "pack_b_vnni": pack_b_vnni(x),
                "pack_b_t_vnni": pack_b_t_vnni(x.T.contiguous()),
            }
            wrong = [
                name for name, out in got.items() if not torch.equal(out, expected)
            ]
            status = "ok" if not wrong else f"WRONG: {', '.join(wrong)}"
            print(f"  {dtype!s:15} {rows}x{cols}: {status}")


Approach = tuple[str, Callable[[torch.Tensor], torch.Tensor]]


def benchmark(
    approaches: list[Approach], dtype: torch.dtype, shapes: list[tuple[int, int]]
) -> None:
    """Time and bandwidth (bytes read + written) of each approach per operand."""
    header = "".join(f"{name:>22}" for name, _ in approaches)
    print(f"  {'shape':>11}{header}")
    for rows, cols in shapes:
        x = torch.randn(rows, cols).to(dtype)
        cells = []
        for _, fn in approaches:
            ms = measure(fn, x)
            moved = (x.numel() + fn(x).numel()) * x.element_size()
            cells.append(f"{ms:8.3f} ms {moved / ms / 1e6:5.1f} GB/s")
        shape = f"{rows}x{cols}"
        print(f"  {shape:>11}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    torch.manual_seed(0)
    shapes = [(4096, 4096), (4090, 4090), (1024, 3008), (1000, 3000)]
    print("Correctness against the layout's definition (aligned and padded):")
    check((torch.float32, torch.bfloat16), [(64, 96), (70, 45), (33, 31)])
    threads = os.environ["OMP_NUM_THREADS"]
    print(f"\nVNNI packing of B [K, N], f32, OMP_NUM_THREADS={threads}:")
    benchmark(
        [
            ("eager", pack_b_vnni_eager),
            ("pack_b_vnni", pack_b_vnni),
            ("pack_b (no VNNI)", pack_b),
            ("pack_b eager", pack_b_eager),
        ],
        torch.float32,
        shapes,
    )
    print("\nVNNI packing of B given as [N, K], f32:")
    benchmark(
        [("eager", pack_b_t_vnni_eager), ("pack_b_t_vnni", pack_b_t_vnni)],
        torch.float32,
        shapes,
    )


if __name__ == "__main__":
    main()
