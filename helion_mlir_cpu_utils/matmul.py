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
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch
from torch import Tensor

import helion_mlir_backend  # noqa: F401

# AMX bf16 register tile. All three extents must divide by this to use the
# blocked path; anything else falls back to torch.matmul.
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
)
def _pack_a_kernel(a4_src: Tensor) -> Tensor:
    """``[M/BM, BM, K/BK, BK]`` -> ``[M/BM, K/BK, BM, BK]``."""
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
    config=helion.Config(block_sizes=[1, 8, 32]),
)
def _pack_b_kernel(b3_src: Tensor) -> Tensor:
    """``[K, N/BN, BN]`` -> ``[N/BN, K, BN]``, one contiguous panel per column block."""
    depth, panels, block_n = b3_src.shape
    out = torch.empty(
        (panels, depth, block_n), dtype=b3_src.dtype, device=b3_src.device
    )
    for panel, tile_k, tile_n in hl.tile([panels, depth, block_n]):
        out[panel, tile_k, tile_n] = b3_src[tile_k, panel, tile_n].permute(1, 0, 2)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def _matmul_blocked_kernel(a4: Tensor, b4: Tensor) -> Tensor:
    """``[MB, KB, BM, BK] x [NB, KB, BK, BN] -> [MB, BM, NB, BN]``."""
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


def pack_a_blocked(a: Tensor) -> Tensor:
    """Pack row-major ``[M, K]`` into ``[M/BM, K/BK, BM, BK]``."""
    m, k = a.shape
    return _pack_a_kernel(
        a.reshape(m // BLOCK_M, BLOCK_M, k // BLOCK_K, BLOCK_K).contiguous()
    )


def pack_b_blocked(b: Tensor) -> Tensor:
    """Pack row-major ``[K, N]`` into ``[N/BN, K/BK, BK, BN]``."""
    k, n = b.shape
    panels = _pack_b_kernel(b.reshape(k, n // BLOCK_N, BLOCK_N).contiguous())
    return panels.view(n // BLOCK_N, k // BLOCK_K, BLOCK_K, BLOCK_N)


def supports(a: Tensor, b: Tensor) -> bool:
    """Whether the blocked Helion path can handle this pair of operands."""
    if a.dtype != b.dtype or a.dtype not in _SUPPORTED_DTYPES:
        return False
    if a.device.type != "cpu" or b.device.type != "cpu":
        return False
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        return False
    m, k = a.shape
    n = b.shape[1]
    return not (m % BLOCK_M or n % BLOCK_N or k % BLOCK_K)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """``A @ B`` for row-major ``[M, K]`` and ``[K, N]``, packing on every call."""
    if not supports(a, b):
        return torch.matmul(a, b)

    m, _ = a.shape
    _, n = b.shape
    out4 = _matmul_blocked_kernel(pack_a_blocked(a), pack_b_blocked(b))
    return out4.view(m, n)
