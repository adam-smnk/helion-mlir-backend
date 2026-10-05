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
tiles instead.

Storing C as ``[M/BM, BM, N/BN, BN]`` rather than ``[M/BM, N/BN, BM, BN]`` is
what removes the separate unpack pass -- the result is reinterpreted as
``[M, N]`` by a metadata-only view.

An optional bias and epilogue are fused into the accumulator before the store
(see ``linear_bf16_blocked_mlir`` in ``benchmarks/helion_mlp_bf16.py`` for the pattern this
follows), so a fused linear+activation costs one kernel instead of a matmul
kernel followed by a separate elementwise kernel.
"""

from __future__ import annotations

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


def _matmul_fused_pack(
    a4: Tensor, b4: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> Tensor:
    """``a4 @ b4`` of row-major operands viewed ``[MB, BM, K/2, 2]`` and
    ``[K/2, 2, NB, BN]`` into ``[MB, BM, NB, BN]``, epilogue fused in.

    Each output tile packs the VNNI B chunk it needs per K step into a private
    buffer and reuses it for all its rows: packed B never leaves the core
    (a separate pack kernel's output is read by many cores, and rewriting
    those lines on the next call costs more than the GEMM saves).
    """
    blocks_m, block_m, pairs, vnni = a4.shape
    pairs2, vnni2, blocks_n, block_n = b4.shape
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
        for tile_pairs in hl.tile(pairs):
            b_vnni = b4[tile_pairs, :, tile_blocks_n, :].permute(2, 0, 3, 1)
            acc = acc + torch.einsum(
                "amcv,bcnv->abmn", a4[tile_blocks_m, :, tile_pairs, :], b_vnni
            )
        y = epilogue(acc)
        out[tile_blocks_m, :, tile_blocks_n, :] = y.permute(0, 2, 1, 3).to(a4.dtype)
    return out


_FUSED_PACK_KERNELS: dict[tuple[int, int, int], helion.Kernel] = {}
_FUSED_PACK_BOUND: dict[tuple[object, ...], BoundKernel] = {}


def _fused_pack_tiles(blocks_m: int, blocks_n: int, pairs: int) -> tuple[int, int, int]:
    """Tile of the tallest column of 4 output blocks (fewest B chunk packs)
    with at least one tile per thread, and up to 1024 K-pairs (64x4x1024 best
    at 4K).

    The f32 accumulator (``tile_m * tile_n`` 4 KiB blocks) and the packed B
    chunk stay within 1 MiB, the stack promotion limit of ``pipeline.yaml``:
    heap buffers per tile cost malloc and page faults.
    """
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    tile_n = _largest_divisor_at_most(blocks_n, 4)
    tile_m = _largest_divisor_at_most(blocks_m, 256 // tile_n)
    while tile_m > 1 and (blocks_m // tile_m) * (blocks_n // tile_n) < threads:
        tile_m = _largest_divisor_at_most(blocks_m, tile_m - 1)
    # AMX VNNI register tiles take 16 K-pairs; a whole-K tile is slower.
    max_k = min(1024, pairs // 2 if pairs >= 32 else pairs)
    tile_k = max(d for d in range(16, max_k + 1, 16) if pairs % d == 0)
    return tile_m, tile_n, tile_k


def _matmul_fused_pack_kernel(
    blocks_m: int, blocks_n: int, pairs: int
) -> helion.Kernel:
    tiles = _fused_pack_tiles(blocks_m, blocks_n, pairs)
    if tiles not in _FUSED_PACK_KERNELS:
        _FUSED_PACK_KERNELS[tiles] = helion.kernel(
            _matmul_fused_pack,
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=list(tiles)),
        )
    return _FUSED_PACK_KERNELS[tiles]


def _matmul_fused_pack_bound(
    a4: Tensor, b4: Tensor, epilogue: Callable[[Tensor], Tensor]
) -> BoundKernel:
    """The fused-pack kernel bound to contiguous ``a4``, ``b4`` and ``epilogue``.

    Cached by problem size and epilogue: the tile choice and Helion's
    specialization lookup (no fast path for callable arguments) cost tens of
    microseconds per call.
    """
    blocks_m, _, pairs, _ = a4.shape
    blocks_n = b4.shape[2]
    key = (blocks_m, blocks_n, pairs, os.environ.get("OMP_NUM_THREADS"), epilogue)
    if (bound := _FUSED_PACK_BOUND.get(key)) is None:
        kernel = _matmul_fused_pack_kernel(blocks_m, blocks_n, pairs)
        bound = _FUSED_PACK_BOUND[key] = kernel.bind((a4, b4, epilogue))
    return bound


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

    if (
        a.dtype == torch.bfloat16
        and not trans_a
        and not trans_b
        and bias is None
        and (m, n, k) == (m_pad, n_pad, k_pad)
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        a4 = a.view(m // BLOCK_M, BLOCK_M, k // 2, 2)
        b4 = b.view(k // 2, 2, n // BLOCK_N, BLOCK_N)
        return _matmul_fused_pack_bound(a4, b4, epilogue)(a4, b4, epilogue).view(m, n)

    if (
        a.dtype == torch.bfloat16
        and not trans_a
        and bias is None
        and (m, k) == (m_pad, k_pad)
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
    """Multiply row-major ``a`` by a RHS produced by ``pack_b_blocked_t``.

    Only the runtime activation is packed on each call. ``n`` is the original
    output width before padding; callers own the lifetime and invalidation of
    ``b4`` and must repack it when the source weight changes.
    """
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
