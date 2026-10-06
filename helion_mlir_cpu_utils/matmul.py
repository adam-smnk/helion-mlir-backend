"""Blocked (mmt4d-style) matmul on Helion's MLIR backend.

The contraction runs on operands in the block layout the AMX lowering expects:

    A = [M/BM, K/BK, BM, BK]
    B = [N/BN, K/BK, BK, BN]
    C = [M/BM, BM, N/BN, BN]   <- a free view of row-major [M, N]

Packing is *not* hoisted out of :func:`matmul`: both operands are packed on every
call, so the work matches what an eager ``torch.matmul`` does with plain
row-major inputs. Each packing kernel copies one 32x32 block per loop iteration
and pads in the same pass: a load past the end of the operand reads zeros.
:func:`pack_b_blocked_vnni` stores B's blocks in the VNNI layout of AMX's bf16
tiles instead. Row-major bf16 operands skip the separate packing:
:func:`_matmul_fused_pack` packs B per output tile and pads only partial
edge tiles.

Storing C as ``[M/BM, BM, N/BN, BN]`` rather than ``[M/BM, N/BN, BM, BN]`` is
what removes the separate unpack pass -- the result is reinterpreted as
``[M, N]`` by a metadata-only view.

An optional bias and epilogue are fused into the accumulator before the store
(see ``linear_bf16_blocked_mlir`` in ``benchmarks/helion_mlp_bf16.py`` for the pattern this
follows), so a fused linear+activation costs one kernel instead of a matmul
kernel followed by a separate elementwise kernel.
"""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import TYPE_CHECKING
from typing import Callable

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

# AMX bf16 register tile. All three extents must divide by this to use the
# blocked path.
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def identity_epilogue(x: Tensor) -> Tensor:
    return x


# Block sizes are literals in the packing loops: Helion cannot resolve a block
# size from a global name.
_PACK_CONFIG = helion.Config(block_sizes=[])


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pad_kernel(x: Tensor, rows: hl.constexpr, cols: hl.constexpr) -> Tensor:
    """``x`` zero-padded to ``[rows, cols]``: tiles past its end read zeros."""
    out = torch.empty((int(rows), int(cols)), dtype=x.dtype, device=x.device)
    for tr, tc in hl.tile([int(rows), int(cols)], block_size=[32, 32]):
        out[tr, tc] = x[tr, tc]
    return out


def pad_2d(x: Tensor, rows: int, cols: int) -> Tensor:
    """Contiguous ``x`` zero-padded to ``[rows, cols]``."""
    return _pad_kernel(x, hl.constexpr(rows), hl.constexpr(cols))


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_a_kernel(a: Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[M, K]`` into ``[M_pad/BM, K_pad/BK, BM, BK]``."""
    mb, kb = int(m_pad) // 32, int(k_pad) // 32
    out = torch.empty((mb, kb, 32, 32), dtype=a.dtype, device=a.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        out[tm.id, tk.id, :, :] = a[tm, tk]
    return out


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_a_kernel_t(a_t: Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr) -> Tensor:
    """Pack transposed ``[K, M]`` into ``[M_pad/BM, K_pad/BK, BM, BK]``."""
    mb, kb = int(m_pad) // 32, int(k_pad) // 32
    out = torch.empty((mb, kb, 32, 32), dtype=a_t.dtype, device=a_t.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        out[tm.id, tk.id, :, :] = a_t[tk, tm].permute(1, 0)
    return out


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_b_kernel(b: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N_pad/BN, K_pad/BK, BK, BN]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b[tk, tn]
    return out


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_b_kernel_t(b_t: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack transposed ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK, BN]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 32, 32), dtype=b_t.dtype, device=b_t.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        out[tn.id, tk.id, :, :] = b_t[tn, tk].permute(1, 0)
    return out


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_b_vnni_kernel(b: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N_pad/BN, K_pad/BK, BK/2, BN, 2]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b.dtype, device=b.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b[tk, tn].reshape(tk.block_size // 2, 2, tn.block_size)
        out[tn.id, tk.id, :, :, :] = pairs.permute(0, 2, 1)
    return out


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_b_vnni_kernel_t(
    b_t: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr
) -> Tensor:
    """Pack transposed ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK/2, BN, 2]``."""
    kb, nb = int(k_pad) // 32, int(n_pad) // 32
    out = torch.empty((nb, kb, 16, 32, 2), dtype=b_t.dtype, device=b_t.device)
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        pairs = b_t[tn, tk].reshape(tn.block_size, tk.block_size // 2, 2)
        out[tn.id, tk.id, :, :, :] = pairs.permute(1, 0, 2)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def _matmul_blocked_kernel(
    a4: Tensor, b4: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """``[MB, KB, BM, BK] x [NB, KB, BK, BN] -> [MB, BM, NB, BN]``, epilogue fused in."""
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
        y = epilogue(acc)
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def _matmul_blocked_kernel_bias(
    a4: Tensor, b4: Tensor, bias3: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """Like ``_matmul_blocked_kernel``, plus a ``[NB, 1, BN]`` bias fused before the epilogue.

    ``bias3`` must stay an explicit kernel argument, not folded into a
    closure captured by ``epilogue``: a closure-captured tensor traces
    incorrectly on this backend (produced garbage, not just stale values, in
    testing) since it never becomes a real kernel input/memref.
    """
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
        y = epilogue(acc + bias3[tile_blocks_n, :, :])
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


def _matmul_vnni(
    a4: Tensor, b4: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """``[MB, BM, K/2, 2] x [NB, K/2, BN, 2] -> [MB, BM, NB, BN]``, epilogue fused in.

    ``a4`` is a free view of row-major A: AMX loads its tiles with A's row stride.
    ``b4`` is B in VNNI panels, the layout of AMX's bf16 B tiles.
    """
    blocks_m, block_m, pairs, vnni = a4.shape
    blocks_n, pairs2, block_n, vnni2 = b4.shape
    assert pairs == pairs2, "K mismatch"
    assert vnni == vnni2, "VNNI factor mismatch"

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
            "amcv,bcnv->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        y = epilogue(acc)
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


_VNNI_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _largest_divisor_at_most(extent: int, limit: int) -> int:
    return max(d for d in range(1, max(1, min(extent, limit)) + 1) if extent % d == 0)


def _vnni_tiles(blocks_m: int, blocks_n: int) -> tuple[int, int]:
    """Output tile, in 32x32 blocks: about one tile per thread, few columns each.

    Every core packing and then reading the same B panels keeps those panels out
    of other cores' caches: B packed into lines that many cores read in the
    previous call costs several times more (invalidations) than the GEMM saves.
    """
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    tile_n = _largest_divisor_at_most(blocks_n, 4)
    per_thread = max(1, blocks_m * (blocks_n // tile_n) // max(1, threads))
    return _largest_divisor_at_most(blocks_m, per_thread), tile_n


def _matmul_vnni_kernel(blocks_m: int, blocks_n: int) -> helion.Kernel:
    tiles = _vnni_tiles(blocks_m, blocks_n)
    if tiles not in _VNNI_KERNELS:
        _VNNI_KERNELS[tiles] = helion.kernel(
            _matmul_vnni,
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=list(tiles)),
        )
    return _VNNI_KERNELS[tiles]


def _vnni_dot(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]`` tiles, ``b`` packed into the VNNI
    layout of AMX's bf16 B tiles (``[K/2, N, 2]``)."""
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0) // 2, 2, b.size(1)).permute(0, 2, 1)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _vnni_dot_t(a: Tensor, b_t: Tensor) -> Tensor:
    """``a @ b_t.T`` of ``[M, K]`` and ``[N, K]`` tiles, ``b_t`` packed into the
    VNNI layout of AMX's bf16 B tiles (``[K/2, N, 2]``)."""
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b_t.reshape(b_t.size(0), b_t.size(1) // 2, 2).permute(1, 0, 2)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _matmul_fused_pack(
    a: Tensor,
    b: Tensor,
    bias: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    k_chunked: hl.constexpr,
    k_even: hl.constexpr,
    trans_b: hl.constexpr,
) -> Tensor:
    """``epilogue(a @ b + bias)`` of row-major ``[M, K]`` and ``[K, N]`` (``[N, K]``
    with ``trans_b``); ``bias`` is ``[N]`` or ``None``.

    Each output tile packs the VNNI B chunk it needs per K step into a private
    buffer and reuses it for all its rows: packed B never leaves the core
    (a separate pack kernel's output is read by many cores, and rewriting
    those lines on the next call costs more than the GEMM saves).

    Tiles past the end of M or N are partial (zero-padded loads, masked
    stores); the compiler runs full tiles without padding. K runs in chunks up
    to ``k_chunked``, one chunk of the rest up to ``k_even`` (a multiple of 64:
    whole AMX steps), then a partial chunk of 64 for the remaining columns.
    """
    m, k = a.shape
    if trans_b:
        n, k2 = b.shape
    else:
        k2, n = b.shape
    assert k == k2, "K mismatch"

    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k_chunked):
            if trans_b:
                acc = acc + _vnni_dot_t(a[tile_m, tile_k], b[tile_n, tile_k])
            else:
                acc = acc + _vnni_dot(a[tile_m, tile_k], b[tile_k, tile_n])
        if k_even > k_chunked:
            for tile_rest in hl.tile(k_chunked, k_even):
                if trans_b:
                    acc = acc + _vnni_dot_t(a[tile_m, tile_rest], b[tile_n, tile_rest])
                else:
                    acc = acc + _vnni_dot(a[tile_m, tile_rest], b[tile_rest, tile_n])
        if k > k_even:
            # A tile as wide as the loop; past K, both operands read zeros.
            for tile_tail in hl.tile(k_even, k_even + 64, block_size=64):
                if trans_b:
                    acc = acc + _vnni_dot_t(a[tile_m, tile_tail], b[tile_n, tile_tail])
                else:
                    acc = acc + _vnni_dot(a[tile_m, tile_tail], b[tile_tail, tile_n])
        if bias is not None:
            acc = acc + bias[tile_n]
        out[tile_m, tile_n] = epilogue(acc).to(a.dtype)
    return out


_FUSED_PACK_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}
# LRU: epilogues created per call must not pile up (each entry pins its epilogue).
_FUSED_PACK_BOUND: OrderedDict[
    tuple[object, ...], tuple[BoundKernel, tuple[hl.constexpr, ...]]
] = OrderedDict()
_FUSED_PACK_BOUND_SIZE = 64
# K elements per chunk: the packed B chunk of a 128-column tile stays within 512 KiB.
_MAX_K_CHUNK = 2048
# Rows of a tile below which packing its B chunks costs more than sharing them.
_MIN_SHARED_ROWS = 256
# Columns of a VNNI panel of :func:`pack_b_vnni_t` (literal in its pack kernel).
_VNNI_PANEL = 64


def _fused_pack_tiles(m: int, n: int, k: int) -> tuple[list[int], int, int]:
    """Block sizes of :func:`_matmul_fused_pack` (rows, columns, K chunk, rest
    chunk), and its ``k_chunked`` and ``k_even``.

    Tiles are columns of up to 128 (fewest B chunk packs), as tall as one tile
    per thread allows (512x128 best at 2K, 2048x128 at 4K); narrower columns
    when that leaves tiles of fewer than ``_MIN_SHARED_ROWS`` rows. The f32 accumulator
    stays within 1 MiB, the stack promotion limit of ``pipeline.yaml``: heap
    buffers per tile cost malloc and page faults. K runs in the fewest chunks of
    at most ``_MAX_K_CHUNK``, and of at most 1 MiB of A when the last row of
    tiles is partial (its A chunks are copied into padded buffers).
    """
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))

    def rows(tile_n: int) -> tuple[int, int]:
        tiles_m = max(1, threads // -(-n // tile_n))
        tile_m = min(
            _round_up(-(-m // tiles_m), BLOCK_M),
            m // BLOCK_M * BLOCK_M,
            (1 << 18) // tile_n,
        )
        return tiles_m, tile_m

    tile_n = min(4 * BLOCK_N, n // BLOCK_N * BLOCK_N)
    tiles_m, tile_m = rows(tile_n)
    # Few rows: narrower columns, each B chunk packed by one tile, not several.
    while tiles_m > 1 and tile_m < _MIN_SHARED_ROWS and tile_n > BLOCK_N:
        tile_n //= 2
        tiles_m, tile_m = rows(tile_n)
    max_chunk = _MAX_K_CHUNK
    if m % tile_m:
        max_chunk = min(max_chunk, max((1 << 19) // tile_m // 64 * 64, 64))
    k_even = k // (2 * BLOCK_K) * 2 * BLOCK_K
    chunks = -(-k_even // max_chunk)
    tile_k = _round_up(-(-k_even // chunks), 2 * BLOCK_K)
    k_chunked = k_even // tile_k * tile_k
    rest = max(k_even - k_chunked, 2 * BLOCK_K)
    return [tile_m, tile_n, tile_k, rest], k_chunked, k_even


def _matmul_fused_pack_bound(
    a: Tensor,
    b: Tensor,
    bias: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    trans_b: bool,
) -> tuple[BoundKernel, tuple[hl.constexpr, ...]]:
    """The fused-pack kernel bound to contiguous ``a``, ``b``, ``bias`` and
    ``epilogue``, with its constexpr arguments.

    Cached by problem size and epilogue: the tile choice and Helion's
    specialization lookup (no fast path for callable arguments) cost tens of
    microseconds per call.
    """
    (m, k), n = a.shape, b.shape[0] if trans_b else b.shape[1]
    key = (m, n, k, trans_b, bias is None, os.environ.get("OMP_NUM_THREADS"), epilogue)
    if (cached := _FUSED_PACK_BOUND.get(key)) is not None:
        _FUSED_PACK_BOUND.move_to_end(key)
        return cached
    block_sizes, k_chunked, k_even = _fused_pack_tiles(m, n, k)
    if (tiles := tuple(block_sizes)) not in _FUSED_PACK_KERNELS:
        _FUSED_PACK_KERNELS[tiles] = helion.kernel(
            _matmul_fused_pack,
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=block_sizes),
        )
    consts = (hl.constexpr(k_chunked), hl.constexpr(k_even), hl.constexpr(trans_b))
    bound = _FUSED_PACK_KERNELS[tiles].bind((a, b, bias, epilogue, *consts))
    cached = _FUSED_PACK_BOUND[key] = (bound, consts)
    if len(_FUSED_PACK_BOUND) > _FUSED_PACK_BOUND_SIZE:
        _FUSED_PACK_BOUND.popitem(last=False)
    return cached


def _matmul_prepacked_vnni(
    a3: Tensor,
    b4: Tensor,
    bias2: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    pairs_chunked: hl.constexpr,
) -> Tensor:
    """``epilogue(a @ b + bias)`` of ``a3``, row-major ``[M, K]`` viewed as K
    pairs ``[M, K/2, 2]``, and ``b4``, B in contiguous column panels of AMX's
    VNNI layout ``[N/P, K/2, P, 2]`` (:func:`pack_b_vnni_t`); ``bias2`` is
    ``[N/P, P]`` or ``None``. Tiles are one panel wide.

    AMX loads both operands' tiles in place: nothing is packed per call. K
    pairs run in chunks up to ``pairs_chunked``, then one chunk of the rest.
    """
    m, pairs, _ = a3.shape
    panels, _, panel, _ = b4.shape

    out = torch.empty((m, panels, panel), dtype=a3.dtype, device=a3.device)
    for tile_m, tile_p in hl.tile([m, panels]):
        acc = hl.zeros([tile_m, tile_p, panel], dtype=torch.float32)
        for tile_kp in hl.tile(pairs_chunked):
            acc = acc + torch.einsum(
                "mcv,bcnv->mbn", a3[tile_m, tile_kp, :], b4[tile_p, tile_kp, :, :]
            )
        if pairs > pairs_chunked:
            for tile_rest in hl.tile(pairs_chunked, pairs):
                acc = acc + torch.einsum(
                    "mcv,bcnv->mbn",
                    a3[tile_m, tile_rest, :],
                    b4[tile_p, tile_rest, :, :],
                )
        if bias2 is not None:
            acc = acc + bias2[tile_p, :]
        out[tile_m, tile_p, :] = epilogue(acc).to(a3.dtype)
    return out


_PREPACKED_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}


def _matmul_prepacked_vnni_call(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    """:func:`matmul_prepacked_b` of a RHS packed by :func:`pack_b_vnni_t`."""
    panels, pairs_b, panel, vnni = map(int, b4.shape)
    m, k = map(int, a.shape)
    n_pad = panels * panel
    if (
        a.dtype != torch.bfloat16
        or b4.dtype != a.dtype
        or (panel, vnni) != (_VNNI_PANEL, 2)
        or k % 2
        or 2 * pairs_b != k
        or _round_up(n, panel) != n_pad
    ):
        raise ValueError(
            f"VNNI-packed RHS {tuple(b4.shape)} {b4.dtype} is incompatible with "
            f"a.shape={tuple(a.shape)} {a.dtype} and n={n}"
        )
    a3 = a.contiguous().view(m, k // 2, 2)
    bias2 = None
    if bias is not None:
        bias2 = (
            bias.reshape(1, n) if n == n_pad else pad_2d(bias.reshape(1, n), 1, n_pad)
        ).view(panels, panel)
    key = (
        "prepacked",
        m,
        n_pad,
        k,
        bias is None,
        os.environ.get("OMP_NUM_THREADS"),
        epilogue,
    )
    if (cached := _FUSED_PACK_BOUND.get(key)) is None:
        threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        tiles_m = max(1, threads // panels)
        tile_m = min(
            _round_up(-(-m // tiles_m), BLOCK_M),
            max(m // BLOCK_M * BLOCK_M, BLOCK_M),
            (1 << 18) // panel,
        )
        chunks = -(-pairs_b // (_MAX_K_CHUNK // 2))
        chunk = min(_round_up(-(-pairs_b // chunks), BLOCK_K), pairs_b)
        pairs_chunked = pairs_b // chunk * chunk
        tiles = (tile_m, 1, chunk, max(pairs_b - pairs_chunked, BLOCK_K))
        if tiles not in _PREPACKED_KERNELS:
            _PREPACKED_KERNELS[tiles] = helion.kernel(
                _matmul_prepacked_vnni,
                static_shapes=True,
                backend="mlir",
                config=helion.Config(block_sizes=list(tiles)),
            )
        consts = (hl.constexpr(pairs_chunked),)
        bound = _PREPACKED_KERNELS[tiles].bind((a3, b4, bias2, epilogue, *consts))
        cached = _FUSED_PACK_BOUND[key] = (bound, consts)
        if len(_FUSED_PACK_BOUND) > _FUSED_PACK_BOUND_SIZE:
            _FUSED_PACK_BOUND.popitem(last=False)
    else:
        _FUSED_PACK_BOUND.move_to_end(key)
    bound, consts = cached
    out = bound(a3, b4, bias2, epilogue, *consts).view(m, n_pad)
    return out if n == n_pad else out[:, :n]


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def _matmul_blocked_kernel_affine(
    a4: Tensor,
    b4: Tensor,
    bias3: Tensor,
    scale3: Tensor,
    post_bias3: Tensor,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    """Blocked matmul with ``(acc + bias) * scale + post_bias`` epilogue."""
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
        y = epilogue(
            (acc + bias3[tile_blocks_n, :, :]) * scale3[tile_blocks_n, :, :]
            + post_bias3[tile_blocks_n, :, :]
        )
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


def pack_a_blocked(
    a: Tensor, m_pad: int | None = None, k_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[M, K]`` into ``[M/BM, K/BK, BM, BK]`` with optional padding."""
    m, k = int(a.shape[0]), int(a.shape[1])
    m_target = _round_up(m, BLOCK_M) if m_pad is None else m_pad
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    return _pack_a_kernel(a, hl.constexpr(m_target), hl.constexpr(k_target))


def pack_a_blocked_t(
    a_t: Tensor, m_pad: int | None = None, k_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[K, M]`` into ``[M/BM, K/BK, BM, BK]`` with optional padding."""
    k, m = int(a_t.shape[0]), int(a_t.shape[1])
    m_target = _round_up(m, BLOCK_M) if m_pad is None else m_pad
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    return _pack_a_kernel_t(a_t, hl.constexpr(m_target), hl.constexpr(k_target))


def pack_b_blocked(
    b: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N/BN, K/BK, BK, BN]`` with optional padding."""
    k, n = int(b.shape[0]), int(b.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_kernel(b, hl.constexpr(k_target), hl.constexpr(n_target))


def pack_b_blocked_t(
    b_t: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK, BN]`` with optional padding."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_kernel_t(b_t, hl.constexpr(k_target), hl.constexpr(n_target))


def pack_b_blocked_vnni(
    b: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[K, N]`` into VNNI blocks ``[N/BN, K/BK, BK/2, BN, 2]``.

    Block ``(j, i)`` is the ``[16, 64]`` bf16 AMX tile of
    ``B[32i:32i+32, 32j:32j+32]``: ``out[j, i, r, n, p] = B[32i + 2r + p, 32j + n]``.
    """
    k, n = int(b.shape[0]), int(b.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_vnni_kernel(b, hl.constexpr(k_target), hl.constexpr(n_target))


def pack_b_blocked_vnni_t(
    b_t: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[N, K]`` (e.g. ``nn.Linear`` weights) into the
    layout of :func:`pack_b_blocked_vnni` of ``b_t.T``."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    return _pack_b_vnni_kernel_t(b_t, hl.constexpr(k_target), hl.constexpr(n_target))


@helion.kernel(static_shapes=True, backend="mlir", config=_PACK_CONFIG)
def _pack_b_vnni_panels_kernel_t(b3_t: Tensor) -> Tensor:
    """``[N, K/2, 2]`` K pairs of transposed B into VNNI panels
    ``[N_pad/64, K/2, 64, 2]`` (``_VNNI_PANEL`` columns, zero-padded)."""
    n, pairs, vnni = b3_t.shape
    panels = -(-n // 64)
    out = torch.empty((panels, pairs, 64, vnni), dtype=b3_t.dtype, device=b3_t.device)
    for tp, tn in hl.tile([pairs, panels * 64], block_size=[16, 64]):
        out[tn.id, tp, :, :] = b3_t[tn, tp, :].permute(1, 0, 2)
    return out


def pack_b_vnni_t(b_t: Tensor) -> Tensor:
    """Pack transposed-layout ``[N, K]`` (e.g. ``nn.Linear`` weights, K even) into
    contiguous column panels of AMX's bf16 VNNI layout of ``b_t.T``,
    ``[N_pad/P, K/2, P, 2]``, for :func:`matmul_prepacked_b`: each output tile
    streams one panel."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    if k % 2:
        raise ValueError(f"pack_b_vnni_t() needs an even K, got b_t.shape={(n, k)}")
    return _pack_b_vnni_panels_kernel_t(b_t.contiguous().view(n, k // 2, 2))


def supports(
    a: Tensor, b: Tensor, trans_a: bool = False, trans_b: bool = False
) -> bool:
    """Whether :func:`matmul` can handle this pair of operands.

    Any 2-D cpu tensor pair of matching dtype/contracted-dim is supported --
    shapes that aren't multiples of the AMX block size are zero-padded up
    to the next multiple internally (lighthouse's own non-Helion matmul
    pipeline handles arbitrary shapes the same way; this mirrors that rather
    than rejecting them).
    """
    if a.dtype != b.dtype or a.dtype not in _SUPPORTED_DTYPES:
        return False
    if a.device.type != "cpu" or b.device.type != "cpu":
        return False
    if a.dim() != 2 or b.dim() != 2:
        return False
    k_a = a.shape[0] if trans_a else a.shape[1]
    k_b = b.shape[1] if trans_b else b.shape[0]
    return k_a == k_b


def _round_up(value: int, block: int) -> int:
    return (value + block - 1) // block * block


def _pad_to(t: Tensor, shape: tuple[int, int]) -> Tensor:
    """Zero-pad a 2-D tensor up to ``shape`` (a no-op if already that shape)."""
    if tuple(t.shape) == shape:
        return t
    padded = torch.zeros(shape, dtype=t.dtype, device=t.device)
    padded[: t.shape[0], : t.shape[1]] = t
    return padded


def matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
) -> Tensor:
    """``epilogue((A.T if trans_a else A) @ (B.T if trans_b else B) + bias)``.

    Packs both operands (and ``bias``, if given) on every call -- no
    pre-packing. ``bias`` is ``[N]``, broadcast over rows. ``epilogue`` is
    fused into the same kernel as the contraction (see module docstring),
    not a separate pass.

    Shapes not divisible by the AMX block size are zero-padded up to the next
    multiple before packing, then the result is sliced back down -- the extra
    padded rows/cols/K-elements are all zero, so they don't affect the real
    output (see :func:`supports`).

    Raises ``ValueError`` if the operands aren't otherwise compatible (mismatched
    dtype/device/rank/contracted-dim -- see :func:`supports`) instead of
    silently falling back to eager PyTorch -- a silent fallback would benchmark
    eager while looking like it benchmarked the Helion kernel. Callers that
    need a fallback must check :func:`supports` themselves and choose one
    explicitly.
    """
    if not supports(a, b, trans_a=trans_a, trans_b=trans_b):
        raise ValueError(
            f"matmul() does not support these operands (trans_a={trans_a}, "
            f"trans_b={trans_b}): a.shape={tuple(a.shape)} a.dtype={a.dtype} "
            f"a.device={a.device}, b.shape={tuple(b.shape)} b.dtype={b.dtype} "
            f"b.device={b.device}. dtype must be one of {_SUPPORTED_DTYPES}, "
            "both tensors must be 2-D on cpu with a matching contracted dim. "
            "Check supports() before calling, or use torch.matmul directly "
            "for unsupported cases."
        )

    m = int(a.shape[1]) if trans_a else int(a.shape[0])
    n = int(b.shape[0]) if trans_b else int(b.shape[1])
    k = int(a.shape[0]) if trans_a else int(a.shape[1])

    m_pad = _round_up(m, BLOCK_M)
    n_pad = _round_up(n, BLOCK_N)
    k_pad = _round_up(k, BLOCK_K)
    # The fused-pack kernel's K steps are pairs of AMX steps of 32.
    k_fused = _round_up(k, 2 * BLOCK_K)

    if a.dtype == torch.bfloat16 and not trans_a:
        if min(m, n) >= BLOCK_M and k >= 2 * BLOCK_K:
            bound, consts = _matmul_fused_pack_bound(a, b, bias, epilogue, trans_b)
            return bound(a, b, bias, epilogue, *consts)
        # Smaller than one tile: padded to one.
        a_p = pad_2d(a, m_pad, k_fused)
        b_p = pad_2d(b, n_pad, k_fused) if trans_b else pad_2d(b, k_fused, n_pad)
        bias_p = None if bias is None else pad_2d(bias.reshape(1, n), 1, n_pad)[0]
        bound, consts = _matmul_fused_pack_bound(a_p, b_p, bias_p, epilogue, trans_b)
        return bound(a_p, b_p, bias_p, epilogue, *consts)[:m, :n]

    if (
        a.dtype == torch.bfloat16
        and not trans_a
        and bias is None
        and (m, k) == (m_pad, k_fused)
        and a.is_contiguous()
    ):
        b4 = (
            pack_b_blocked_vnni_t(b, k_pad=k_pad, n_pad=n_pad)
            if trans_b
            else pack_b_blocked_vnni(b, k_pad=k_pad, n_pad=n_pad)
        )
        a4 = a.view(m // BLOCK_M, BLOCK_M, k // 2, 2)
        b4 = b4.view(n_pad // BLOCK_N, k // 2, BLOCK_N, 2)
        out4 = _matmul_vnni_kernel(m // BLOCK_M, n_pad // BLOCK_N)(a4, b4, epilogue)
        return out4.reshape(m, n_pad)[:, :n]

    a4 = (
        pack_a_blocked_t(a, m_pad=m_pad, k_pad=k_pad)
        if trans_a
        else pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    )
    b4 = (
        pack_b_blocked_t(b, k_pad=k_pad, n_pad=n_pad)
        if trans_b
        else pack_b_blocked(b, k_pad=k_pad, n_pad=n_pad)
    )
    if bias is None:
        out4 = _matmul_blocked_kernel(a4, b4, epilogue)
    else:
        bias_padded = _pad_to(bias.reshape(1, -1), (1, n_pad)).reshape(-1)
        bias3 = bias_padded.reshape(n_pad // BLOCK_N, 1, BLOCK_N)
        out4 = _matmul_blocked_kernel_bias(a4, b4, bias3, epilogue)
    out = out4.reshape(m_pad, n_pad)
    return out[:m, :n]


def matmul_prepacked_b(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
) -> Tensor:
    """Multiply row-major ``a`` by a RHS produced by ``pack_b_blocked_t`` or, for
    bf16, :func:`pack_b_vnni_t`.

    Only the runtime activation is packed on each call (not even that for a
    VNNI-packed RHS). ``n`` is the original output width before padding;
    callers own the lifetime and invalidation of ``b4`` and must repack it when
    the source weight changes.
    """
    if a.dim() == 2 and b4.dim() == 4 and b4.shape[-1] == 2:
        return _matmul_prepacked_vnni_call(a, b4, n, bias, epilogue)
    if a.dim() != 2 or b4.dim() != 4:
        raise ValueError(
            f"matmul_prepacked_b() expects rank-2 a and rank-4 b4, got "
            f"a.shape={tuple(a.shape)} and b4.shape={tuple(b4.shape)}"
        )
    if a.dtype != b4.dtype or a.device != b4.device:
        raise ValueError(
            "matmul_prepacked_b() requires matching dtype and device: "
            f"a=({a.dtype}, {a.device}), b4=({b4.dtype}, {b4.device})"
        )

    blocks_n, blocks_k, block_k, block_n = map(int, b4.shape)
    if block_k != BLOCK_K or block_n != BLOCK_N:
        raise ValueError(
            f"invalid packed RHS block shape {tuple(b4.shape)}; "
            f"expected trailing dimensions ({BLOCK_K}, {BLOCK_N})"
        )

    m, k = map(int, a.shape)
    m_pad = _round_up(m, BLOCK_M)
    k_pad = _round_up(k, BLOCK_K)
    n_pad = blocks_n * BLOCK_N
    if (
        blocks_k * BLOCK_K != k_pad
        or not 0 < n <= n_pad
        or _round_up(n, BLOCK_N) != n_pad
    ):
        raise ValueError(
            f"packed RHS shape {tuple(b4.shape)} is incompatible with "
            f"a.shape={tuple(a.shape)} and n={n}"
        )

    a4 = pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    if bias is None:
        out4 = _matmul_blocked_kernel(a4, b4, epilogue)
    else:
        if bias.dim() != 1 or int(bias.shape[0]) != n:
            raise ValueError(f"bias must have shape ({n},), got {tuple(bias.shape)}")
        bias_padded = _pad_to(bias.reshape(1, -1), (1, n_pad)).reshape(-1)
        bias3 = bias_padded.reshape(blocks_n, 1, BLOCK_N)
        out4 = _matmul_blocked_kernel_bias(a4, b4, bias3, epilogue)
    return out4.reshape(m_pad, n_pad)[:m, :n]


def matmul_prepacked_b_affine(
    a: Tensor,
    b4: Tensor,
    n: int,
    bias: Tensor | None,
    post_scale: Tensor,
    post_bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
) -> Tensor:
    """Prepacked RHS matmul with a per-output affine post-op."""
    blocks_n, blocks_k, block_k, block_n = map(int, b4.shape)
    m, k = map(int, a.shape)
    m_pad = _round_up(m, BLOCK_M)
    k_pad = _round_up(k, BLOCK_K)
    n_pad = blocks_n * BLOCK_N
    if (
        a.dim() != 2
        or b4.dim() != 4
        or a.dtype != b4.dtype
        or a.device != b4.device
        or block_k != BLOCK_K
        or block_n != BLOCK_N
        or blocks_k * BLOCK_K != k_pad
        or not 0 < n <= n_pad
        or _round_up(n, BLOCK_N) != n_pad
    ):
        raise ValueError(
            f"packed RHS shape {tuple(b4.shape)} is incompatible with "
            f"a.shape={tuple(a.shape)} and n={n}"
        )

    def prepare(vector: Tensor | None, fill: float) -> Tensor:
        if vector is None:
            return torch.full((n_pad,), fill, dtype=a.dtype, device=a.device)
        if vector.dim() > 1 or (
            vector.dim() == 1 and int(vector.shape[0]) not in (1, n)
        ):
            raise ValueError(
                f"affine vector must be scalar or have shape (1,) or ({n},), "
                f"got {tuple(vector.shape)}"
            )
        vector = vector.to(dtype=a.dtype, device=a.device)
        if vector.numel() == 1:
            return vector.expand(n_pad).contiguous()
        return _pad_to(vector.reshape(1, -1), (1, n_pad)).reshape(-1)

    a4 = pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    bias3 = prepare(bias, 0.0).reshape(blocks_n, 1, BLOCK_N)
    scale3 = prepare(post_scale, 1.0).reshape(blocks_n, 1, BLOCK_N)
    post_bias3 = prepare(post_bias, 0.0).reshape(blocks_n, 1, BLOCK_N)
    out4 = _matmul_blocked_kernel_affine(a4, b4, bias3, scale3, post_bias3, epilogue)
    return out4.reshape(m_pad, n_pad)[:m, :n]


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """Batched ``A @ B``; each batch slice is packed and contracted independently.

    A genuine single-kernel batched version (one extra leading tile dimension
    threaded through the pack + contract kernels) was prototyped and measured
    ~30% *slower* than this per-batch-call loop at BATCH=3, M=N=K=2048 -- the
    combined kernel gets
    worse thread-level parallelism across the batch dimension on this backend
    than launching one fully-parallel kernel per batch slice. Kept as the
    simpler, faster loop.
    """
    return torch.stack([matmul(a[i], b[i]) for i in range(a.shape[0])])
