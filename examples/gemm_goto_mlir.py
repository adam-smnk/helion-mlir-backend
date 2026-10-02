"""
Goto-style GEMM on the MLIR Backend

GotoBLAS-style GEMMs (BLIS, oneDNN) nest their loops so that each level of the
cache holds one operand panel: a ``KC x NC`` panel of B stays in L2 while
``MC x KC`` panels of A stream through it, and the micro-kernel keeps an
``MR x NR`` tile of C in registers. In Helion that nesting is written with a
loop over one tile of an outer loop, ``hl.tile(t.begin, t.end)``:

- ``gemm_goto``: each parallel ``MO x NC`` tile of C is split into ``MC``-row
  panels; each panel accumulates over all of K, reading ``KC``-deep slices of A
  and of the tile's B columns.
- ``gemm_goto_inplace``: the loop order of the textbook algorithm. For every
  ``KC`` step one ``KC x NC`` panel of B is loaded once and reused by every
  ``MC``-row panel of A in the tile, which updates C in place.

Iterations of the outer loop write disjoint tiles of C, and each nested panel
lies inside its tile, so the outer loop runs in parallel. When the block sizes
divide each other and the shapes, every tile is full and the IR is static.

The checks compare against ``torch.matmul`` on aligned, padded and odd shapes,
and the timings are a quick f32 ballpark against eager PyTorch and a plain
Helion matmul on row-major tiles. Shapes that the tiles do not divide take the
masked path for their partial tiles (see ``gemm_fused_packing_mlir.py``, which
pads while packing). Run with ``LD_PRELOAD=libtcmalloc.so.4`` to keep large
outputs from being page-faulted in on every call.
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

import helion_mlir_backend  # noqa: F401


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    # MO, NC (parallel tile of C), MC (rows of an A panel), KC (K step).
    config=helion.Config(block_sizes=[128, 512, 64, 64]),
)
def gemm_goto(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b``: row panels of each C tile, each accumulated over all of K.

    One parallel iteration, for the ``MO x NC`` tile of C at ``(tile_m, tile_n)``::

                                    b[:, tile_n]
                                    +-----------+
                                    |   k = 0   |  KC x NC
                                    +-----------+
                                    |   k = 1   |
                                    +-----------+
                                    |    ...    |
                                    +-----------+
               a[tile_m, :]          C tile (MO x NC)
            +-----+-----+-----+     +-----------+
         MC | k=0 | k=1 | ... | --> |  panel 0  |  acc = sum_k a[panel, k] @ b[k, tile_n]
            +-----+-----+-----+     +-----------+
         MC | k=0 | k=1 | ... | --> |  panel 1  |
            +-----+-----+-----+     +-----------+

    Panels run one after another, each sweeping all of K into an ``MC x NC``
    accumulator that is stored once: panel 0 (k = 0, 1, ...), then panel 1
    (k = 0, 1, ...). Every panel re-reads all of the tile's B columns.
    """
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        for panel_m in hl.tile(tile_m.begin, tile_m.end):
            acc = hl.zeros([panel_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[panel_m, tile_k], b[tile_k, tile_n])
            out[panel_m, tile_n] = acc
    return out


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    # MO, NC (parallel tile of C), KC (depth of a B panel), MC (rows of an A panel).
    config=helion.Config(block_sizes=[256, 256, 256, 64]),
)
def gemm_goto_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b``: per K step, one B panel reused by every A panel of the C tile.

    One parallel iteration, for the ``MO x NC`` tile of C at ``(tile_m, tile_n)``,
    at one K step ``tile_k``::

                                    panel_b = b[tile_k, tile_n]
                                    +-----------+
                                    |///////////|  KC x NC, loaded once
                                    +-----------+
             a[tile_m, tile_k]       C tile (MO x NC)
                  +-----+           +-----------+
               MC |  0  |  ------>  |  panel 0  |  += a[panel 0, tile_k] @ panel_b
                  +-----+           +-----------+
               MC |  1  |  ------>  |  panel 1  |  += a[panel 1, tile_k] @ panel_b
                  +-----+           +-----------+
                  | ... |           |    ...    |
                  +-----+           +-----------+

    K is the outer loop: k = 0 (panels 0, 1, ...), then k = 1 (panels 0, 1,
    ...). The B panel stays hot while every A panel streams through it, at the
    cost of reading and writing the C tile on every K step.
    """
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        for tile_k in hl.tile(k):
            panel_b = b[tile_k, tile_n]
            for panel_m in hl.tile(tile_m.begin, tile_m.end):
                out[panel_m, tile_n] = torch.addmm(
                    out[panel_m, tile_n], a[panel_m, tile_k], panel_b
                )
    return out


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


APPROACHES: list[tuple[str, Callable[..., torch.Tensor]]] = [
    ("gemm_goto", gemm_goto),
    ("gemm_goto_inplace", gemm_goto_inplace),
    ("gemm_plain", gemm_plain),
]


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
            for name, fn in APPROACHES
            if not torch.allclose(fn(a, b), expected, atol=1e-3, rtol=1e-3)
        ]
        print(f"  {m}x{k}x{n}: {'ok' if not wrong else 'WRONG: ' + ', '.join(wrong)}")


def benchmark(shapes: list[tuple[int, int, int]]) -> None:
    approaches = [("torch", torch.matmul), *APPROACHES]
    print(f"  {'M x K x N':>16}" + "".join(f"{name:>22}" for name, _ in approaches))
    for m, k, n in shapes:
        a, b = torch.randn(m, k), torch.randn(k, n)
        cells = []
        for _, fn in approaches:
            ms = measure(fn, a, b)
            cells.append(f"{ms:7.2f} ms {2 * m * n * k / ms / 1e6:5.0f} GF/s")
        shape = f"{m}x{k}x{n}"
        print(f"  {shape:>16}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    torch.manual_seed(0)
    print("Correctness against torch.matmul (f32):")
    check([(256, 128, 512), (70, 45, 50), (300, 200, 600)])
    print(f"\nf32 GEMM, OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}:")
    benchmark([(2048, 2048, 2048), (1024, 3072, 1536), (1000, 3000, 1500)])


if __name__ == "__main__":
    main()
