"""
Batch Matrix Multiplication with MLIR Backend

Demonstrates a tiled batch matrix multiplication kernel:
    out[b, m, n] = A[b, m, k] @ B[b, k, n]

This example is CPU-focused and uses float32 inputs for portability.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from helion_mlir_backend import generate_mlir


@helion.kernel(static_shapes=True)
def bmm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tiled dense batch matrix multiplication."""
    batch, m, k = a.shape
    batch2, k2, n = b.shape
    assert batch == batch2
    assert k == k2

    out = torch.zeros((batch, m, n), dtype=torch.float32, device=a.device)

    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc

    return out


def main() -> None:
    """Generate and print MLIR for the BMM kernel."""
    device = torch.device("cpu")
    batch, m, k, n = 4, 32, 48, 24

    a = torch.randn(batch, m, k, dtype=torch.float32, device=device)
    b = torch.randn(batch, k, n, dtype=torch.float32, device=device)

    print("=" * 80)
    print("BMM MLIR Generation (CPU)")
    print("=" * 80)
    print(f"Input shapes: a={a.shape}, b={b.shape}")
    print(f"Output shape: ({batch}, {m}, {n})")

    module = generate_mlir(bmm_kernel, [a, b])
    print("\nGenerated MLIR:")
    print("=" * 80)
    print(str(module))


if __name__ == "__main__":
    main()
