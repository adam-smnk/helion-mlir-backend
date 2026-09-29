"""
Block Packing for Blocked Matmuls with the MLIR Backend

Blocked (mmt4d-style) matmuls, such as the AMX path in
``helion_mlir_cpu_utils.matmul``, read their operands in 32x32 blocks:

    A [M, K] -> [M/32, K/32, 32, 32]   block (i, j) holds A[32i:32i+32, 32j:32j+32]
    B [K, N] -> [N/32, K/32, 32, 32]   block (j, i) holds B[32i:32i+32, 32j:32j+32]

Extents that are not multiples of 32 are zero-padded. This example compares four
ways to pack B, checks them against each other, and times them:

1. PyTorch eager: ``pad`` + ``reshape`` + ``permute`` + ``contiguous``.
2. Host padding + Helion kernel: the previous ``helion_mlir_cpu_utils`` kernel.
   The kernel's host code pads B with ``torch.zeros`` and a slice copy, so a
   padded operand is read and written twice; the device loop moves panels.
3. Inline MLIR ``linalg.pack``: one ``linalg.pack`` of the whole operand with a
   zero ``padding_value``, the op lighthouse's ``block-pack-matmul`` produces.
   The pipeline's pack lowering tiles, vectorizes and parallelizes it.
4. Helion, all in one kernel: each iteration of the loop over the packed blocks
   copies one 32x32 block, ``out[tn.id, tk.id, :, :] = b[tk, tn]``. A load past
   the end of B reads zeros, so padding is part of the same pass, and a block B
   covers entirely is a plain copy.

``pack_a``, ``pack_b``, ``pack_a_t`` and ``pack_b_t`` (operands given
transposed, e.g. ``nn.Linear`` weights) are the resulting kernels. The
transposed ones pad before transposing, a slower path: on small padded operands
they can fall behind eager.

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

from helion_mlir_backend import inline_mlir

BLOCK = 32
CONFIG = helion.Config(block_sizes=[], mlir_pipeline="opt")


# ---------------------------------------------------------------------------
# The kernels: all the work, padding included, in one Helion kernel. Block
# sizes are literals: Helion cannot resolve a block size from a global name.
# ---------------------------------------------------------------------------


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_a(a: torch.Tensor) -> torch.Tensor:
    """``[M, K] -> [M/32, K/32, 32, 32]``, zero-padded."""
    m, k = a.shape
    mb, kb = (m + 31) // 32, (k + 31) // 32
    out = torch.empty((mb, kb, 32, 32), dtype=a.dtype, device=a.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        out[tm.id, tk.id, :, :] = a[tm, tk]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_b(b: torch.Tensor) -> torch.Tensor:
    """``[K, N] -> [N/32, K/32, 32, 32]``, zero-padded."""
    k, n = b.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b[tk, tn]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_a_t(a_t: torch.Tensor) -> torch.Tensor:
    """A given as ``[K, M]``: the same layout as ``pack_a(a_t.T)``."""
    k, m = a_t.shape
    mb, kb = (m + 31) // 32, (k + 31) // 32
    out = torch.empty((mb, kb, 32, 32), dtype=a_t.dtype, device=a_t.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        out[tm.id, tk.id, :, :] = a_t[tk, tm].permute(1, 0)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_b_t(b_t: torch.Tensor) -> torch.Tensor:
    """B given as ``[N, K]`` (e.g. ``nn.Linear`` weights): as ``pack_b(b_t.T)``."""
    n, k = b_t.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b_t.dtype, device=b_t.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b_t[tn, tk].permute(1, 0)
    return out


# ---------------------------------------------------------------------------
# The alternatives, for comparison.
# ---------------------------------------------------------------------------


def _pad(x: torch.Tensor) -> torch.Tensor:
    rows, cols = (-x.shape[0]) % BLOCK, (-x.shape[1]) % BLOCK
    return torch.nn.functional.pad(x, (0, cols, 0, rows))


def pack_a_eager(a: torch.Tensor) -> torch.Tensor:
    padded = _pad(a)
    mb, kb = padded.shape[0] // BLOCK, padded.shape[1] // BLOCK
    return padded.reshape(mb, BLOCK, kb, BLOCK).permute(0, 2, 1, 3).contiguous()


def pack_b_eager(b: torch.Tensor) -> torch.Tensor:
    padded = _pad(b)
    kb, nb = padded.shape[0] // BLOCK, padded.shape[1] // BLOCK
    return padded.reshape(kb, BLOCK, nb, BLOCK).permute(2, 0, 1, 3).contiguous()


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 4096], mlir_pipeline="opt"),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pack_b_host_padded(b: torch.Tensor) -> torch.Tensor:
    """The previous kernel: pad on the host, then move ``[K, 32]`` panels."""
    k, n = b.shape
    depth, panels = (k + 31) // 32 * 32, (n + 31) // 32
    if k == depth and n == panels * 32:
        b3 = b.reshape(depth, panels, 32)
    else:
        padded = torch.zeros((depth, panels * 32), dtype=b.dtype, device=b.device)
        padded[:k, :n] = b
        b3 = padded.reshape(depth, panels, 32)
    out = torch.empty((panels, depth, 32), dtype=b.dtype, device=b.device)
    for panel in hl.tile(panels):
        for tile_k in hl.tile(depth):
            out[panel, tile_k, :] = b3[tile_k, panel, :].permute(1, 0, 2)
    return out


def _linalg_pack_b(element: str) -> str:
    return f"""
func.func @pack_b(%b: tensor<?x?x{element}>, %dest: tensor<?x?x32x32x{element}>)
    -> tensor<?x?x32x32x{element}> {{
  %zero = arith.constant 0.0 : {element}
  %packed = linalg.pack %b padding_value(%zero : {element})
      outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
      into %dest : tensor<?x?x{element}> -> tensor<?x?x32x32x{element}>
  return %packed : tensor<?x?x32x32x{element}>
}}
"""


# Helion only indexes dicts by int or str.
LINALG_PACK_B = {
    str(torch.float32): _linalg_pack_b("f32"),
    str(torch.bfloat16): _linalg_pack_b("bf16"),
}


@helion.kernel(backend="mlir", static_shapes=True, config=CONFIG)
def pack_b_linalg(b: torch.Tensor) -> torch.Tensor:
    """One ``linalg.pack`` of all of B, written straight into the output."""
    k, n = b.shape
    kb, nb = (k + 31) // 32, (n + 31) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    for _ in hl.grid(1):
        whole = out[:, :, :, :]
        out[:, :, :, :] = inline_mlir(
            LINALG_PACK_B[str(b.dtype)], [b[:, :], whole], whole
        )
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
            expected = {
                "pack_a": pack_a_eager(x),
                "pack_b": pack_b_eager(x),
                "pack_a_t": pack_a_eager(x.T),
                "pack_b_t": pack_b_eager(x.T),
            }
            got = {
                "pack_a": pack_a(x),
                "pack_b": pack_b(x),
                "pack_a_t": pack_a_t(x),
                "pack_b_t": pack_b_t(x),
            }
            b_shape = expected["pack_b"].shape
            got["host padded"] = pack_b_host_padded(x).reshape(b_shape)
            expected["host padded"] = expected["pack_b"]
            got["linalg.pack"] = pack_b_linalg(x)
            expected["linalg.pack"] = expected["pack_b"]
            wrong = [name for name in got if not torch.equal(got[name], expected[name])]
            status = "ok" if not wrong else f"WRONG: {', '.join(wrong)}"
            print(f"  {dtype!s:15} {rows}x{cols}: {status}")


Approach = tuple[str, Callable[[torch.Tensor], torch.Tensor]]


def benchmark(
    approaches: list[Approach],
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int]],
) -> None:
    """Time and bandwidth (bytes read + written) of each approach per operand."""
    header = "".join(f"{name:>22}" for name, _ in approaches)
    print(f"  {'dtype':15}{'shape':>11}{header}")
    for dtype in dtypes:
        for rows, cols in shapes:
            x = torch.randn(rows, cols).to(dtype)
            cells = []
            for _, fn in approaches:
                ms = measure(fn, x)
                moved = (x.numel() + fn(x).numel()) * x.element_size()
                cells.append(f"{ms:8.3f} ms {moved / ms / 1e6:5.1f} GB/s")
            shape = f"{rows}x{cols}"
            print(f"  {dtype!s:15}{shape:>11}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    torch.manual_seed(0)
    dtypes = (torch.float32, torch.bfloat16)
    shapes = [(4096, 4096), (4090, 4090), (1024, 3008), (1000, 3000)]
    print("Correctness against eager PyTorch (aligned and padded):")
    check(dtypes, [(96, 64), (70, 45)])
    print(f"\nPacking B [K, N], OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}:")
    benchmark(
        [
            ("eager", pack_b_eager),
            ("host pad + Helion", pack_b_host_padded),
            ("inline linalg.pack", pack_b_linalg),
            ("Helion kernel", pack_b),
        ],
        dtypes,
        shapes,
    )
    print("\nThe other layouts, Helion kernel vs eager:")
    benchmark(
        [
            ("pack_a", pack_a),
            ("pack_a eager", pack_a_eager),
            ("pack_a_t", pack_a_t),
            ("pack_a_t eager", lambda a_t: pack_a_eager(a_t.T)),
            ("pack_b_t", pack_b_t),
            ("pack_b_t eager", lambda b_t: pack_b_eager(b_t.T)),
        ],
        dtypes,
        shapes,
    )


if __name__ == "__main__":
    main()
