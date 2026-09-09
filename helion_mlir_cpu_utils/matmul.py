"""Blocked (mmt4d-style) matmul on Helion's MLIR backend.

The contraction runs on operands in the block layout the AMX lowering expects:

    A = [M/BM, K/BK, BM, BK]
    B = [N/BN, K/BK, BK, BN]
    C = [M/BM, BM, N/BN, BN]   <- a free view of row-major [M, N]

Packing is *not* hoisted out of :func:`matmul`: both operands are packed on every
call, so the work matches what an eager ``torch.matmul`` does with plain
row-major inputs.

Storing C as ``[M/BM, BM, N/BN, BN]`` rather than ``[M/BM, N/BN, BM, BN]`` is
what removes the separate unpack pass -- the result is reinterpreted as
``[M, N]`` by a metadata-only view.

An optional bias and epilogue are fused into the accumulator before the store
(see ``linear_bf16_blocked_mlir`` in ``helion_mlp_bf16.py`` for the pattern this
follows), so a fused linear+activation costs one kernel instead of a matmul
kernel followed by a separate elementwise kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

# AMX bf16 register tile. All three extents must divide by this to use the
# blocked path.
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def identity_epilogue(x: Tensor) -> Tensor:
    return x


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _pack_a_kernel(a: Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[M, K]`` into ``[M_pad/BM, K_pad/BK, BM, BK]``."""
    m, k = int(a.shape[0]), int(a.shape[1])
    bm, bk = int(m_pad) // 32, int(k_pad) // 32
    if m == int(m_pad) and k == int(k_pad):
        a4 = a.reshape(bm, 32, bk, 32)
    else:
        pad = torch.zeros((int(m_pad), int(k_pad)), dtype=a.dtype, device=a.device)
        pad[:m, :k] = a
        a4 = pad.reshape(bm, 32, bk, 32)

    out = torch.empty((bm, bk, 32, 32), dtype=a.dtype, device=a.device)
    for bmi, bki, tm, tk in hl.tile([bm, bk, 32, 32]):
        out[bmi, bki, tm, tk] = a4[bmi, tm, bki, tk].permute(0, 2, 1, 3)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 4096]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _pack_b_kernel(b: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N_pad/BN, K_pad, BN]``."""
    k, n = int(b.shape[0]), int(b.shape[1])
    depth = int(k_pad)
    panels = int(n_pad) // 32
    if k == int(k_pad) and n == int(n_pad):
        b3 = b.reshape(depth, panels, 32)
    else:
        pad = torch.zeros((int(k_pad), int(n_pad)), dtype=b.dtype, device=b.device)
        pad[:k, :n] = b
        b3 = pad.reshape(depth, panels, 32)

    out = torch.empty((panels, depth, 32), dtype=b.dtype, device=b.device)
    # Nested (not combined) tile loops: each block-count dim gets its own
    # ragged-safe mask and a much larger per-iteration chunk than tiling the
    # raw depth extent directly (previously ~65k tiny 8x32 iterations).
    # The public wrapper rewraps this function with exact-divisor panel/depth
    # blocks for each padded shape, avoiding unsafe ragged stores.
    for panel in hl.tile(panels):
        for tile_k in hl.tile(depth):
            out[panel, tile_k, :] = b3[tile_k, panel, :].permute(1, 0, 2)
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


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _pack_a_kernel_t(a_t: Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr) -> Tensor:
    """Pack transposed ``[K, M]`` into ``[M_pad/BM, K_pad/BK, BM, BK]``."""
    k, m = int(a_t.shape[0]), int(a_t.shape[1])
    bm, bk = int(m_pad) // 32, int(k_pad) // 32
    if m == int(m_pad) and k == int(k_pad):
        a4 = a_t.reshape(bk, 32, bm, 32)
    else:
        pad = torch.zeros((int(k_pad), int(m_pad)), dtype=a_t.dtype, device=a_t.device)
        pad[:k, :m] = a_t
        a4 = pad.reshape(bk, 32, bm, 32)

    out = torch.empty((bm, bk, 32, 32), dtype=a_t.dtype, device=a_t.device)
    for bmi, bki, tm, tk in hl.tile([bm, bk, 32, 32]):
        out[bmi, bki, tm, tk] = a4[bki, tk, bmi, tm].permute(2, 0, 3, 1)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 4096]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _pack_b_kernel_t(b_t: Tensor, k_pad: hl.constexpr, n_pad: hl.constexpr) -> Tensor:
    """Pack transposed ``[N, K]`` into ``[N_pad/BN, K_pad, BN]``."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    depth = int(k_pad)
    panels = int(n_pad) // 32
    if k == int(k_pad) and n == int(n_pad):
        b3 = b_t.reshape(panels, 32, depth)
    else:
        pad = torch.zeros((int(n_pad), int(k_pad)), dtype=b_t.dtype, device=b_t.device)
        pad[:n, :k] = b_t
        b3 = pad.reshape(panels, 32, depth)

    out = torch.empty((panels, depth, 32), dtype=b_t.dtype, device=b_t.device)
    # See _pack_b_kernel: depth block intentionally oversized (safe on this
    # inner nested dim), panel block stays 1 (outer/store-position dim).
    for panel in hl.tile(panels):
        for tile_k in hl.tile(depth):
            out[panel, tile_k, :] = b3[panel, :, tile_k].permute(0, 2, 1)
    return out


# Seed the shape-specialized caches with the common MLP configuration.
_pack_b_kernel_wide = helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[8, 4096]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)(_pack_b_kernel.fn)
_pack_b_kernel_t_wide = helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[8, 4096]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)(_pack_b_kernel_t.fn)

_pack_b_kernel_configs = {(8, 4096): _pack_b_kernel_wide}
_pack_b_kernel_t_configs = {(8, 4096): _pack_b_kernel_t_wide}


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


def _pack_b_block_sizes(n_target: int, k_target: int) -> tuple[int, int]:
    panels = n_target // BLOCK_N
    panel_block = next(
        (candidate for candidate in range(8, 1, -1) if panels % candidate == 0),
        panels,
    )
    return panel_block, k_target


def _pack_b_kernel_for_shape(n_target: int, k_target: int, transposed: bool) -> Kernel:
    block_sizes = _pack_b_block_sizes(n_target, k_target)
    configs = _pack_b_kernel_t_configs if transposed else _pack_b_kernel_configs
    if block_sizes not in configs:
        source = _pack_b_kernel_t if transposed else _pack_b_kernel
        configs[block_sizes] = helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=list(block_sizes)),
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )(source.fn)
    return configs[block_sizes]


def pack_b_blocked(
    b: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N/BN, K/BK, BK, BN]`` with optional padding."""
    k, n = int(b.shape[0]), int(b.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    pack_kernel = _pack_b_kernel_for_shape(n_target, k_target, transposed=False)
    panels = pack_kernel(b, hl.constexpr(k_target), hl.constexpr(n_target))
    return panels.view(n_target // BLOCK_N, k_target // BLOCK_K, BLOCK_K, BLOCK_N)


def pack_b_blocked_t(
    b_t: Tensor, k_pad: int | None = None, n_pad: int | None = None
) -> Tensor:
    """Pack transposed-layout ``[N, K]`` into ``[N_pad/BN, K_pad/BK, BK, BN]`` with optional padding."""
    n, k = int(b_t.shape[0]), int(b_t.shape[1])
    k_target = _round_up(k, BLOCK_K) if k_pad is None else k_pad
    n_target = _round_up(n, BLOCK_N) if n_pad is None else n_pad
    pack_kernel = _pack_b_kernel_for_shape(n_target, k_target, transposed=True)
    panels = pack_kernel(b_t, hl.constexpr(k_target), hl.constexpr(n_target))
    return panels.view(n_target // BLOCK_N, k_target // BLOCK_K, BLOCK_K, BLOCK_N)


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
    ~30% *slower* than this per-batch-call loop at BATCH=3, M=N=K=2048 (see
    temp/probe_bmm_bench.py in the repo history) -- the combined kernel gets
    worse thread-level parallelism across the batch dimension on this backend
    than launching one fully-parallel kernel per batch slice. Kept as the
    simpler, faster loop.
    """
    return torch.stack([matmul(a[i], b[i]) for i in range(a.shape[0])])
