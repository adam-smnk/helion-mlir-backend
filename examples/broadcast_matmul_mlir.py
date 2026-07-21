"""
Broadcast Batch Matmul with MLIR Backend

Target operation:
    X[B, M, K] @ W[K, N] -> Out[B, M, N]

For current MLIR backend compatibility, the broadcasted weight is materialized
on the host as Wb[B, K, N] before kernel invocation. The kernel then performs
standard tiled batch matmul with torch.baddbmm.

This example is CPU-focused and uses float32 inputs.
"""

from __future__ import annotations

import helion
import helion.language as hl
from helion.mlir import generate_mlir
import torch


@helion.kernel(static_shapes=True)
def broadcast_matmul_kernel(x: torch.Tensor, w_batched: torch.Tensor) -> torch.Tensor:
    """Batch matmul where broadcasted weights are already materialized."""
    b, m, k = x.shape
    b2, k2, n = w_batched.shape
    assert b == b2
    assert k == k2

    out = torch.zeros((b, m, n), dtype=torch.float32, device=x.device)

    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc,
                x[tile_b, tile_m, tile_k],
                w_batched[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = acc

    return out


def main() -> None:
    """Generate and print MLIR for broadcast batch matmul."""
    device = torch.device("cpu")
    b, m, k, n = 4, 32, 48, 24

    x = torch.randn(b, m, k, dtype=torch.float32, device=device)
    w = torch.randn(k, n, dtype=torch.float32, device=device)
    w_batched = w.unsqueeze(0).expand((b, k, n)).contiguous()

    print("=" * 80)
    print("Broadcast Matmul MLIR Generation (CPU)")
    print("=" * 80)
    print(f"Input shapes: x={x.shape}, w={w.shape}, w_batched={w_batched.shape}")
    print(f"Output shape: ({b}, {m}, {n})")

    module = generate_mlir(broadcast_matmul_kernel, [x, w_batched])
    print("\nGenerated MLIR:")
    print("=" * 80)
    print(str(module))


if __name__ == "__main__":
    main()
