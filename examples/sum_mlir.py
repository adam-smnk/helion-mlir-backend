"""
Row-Wise Sum Reduction with MLIR Backend

Demonstrates reduction over the last dimension of a 2D tensor:
    out[m] = sum_n x[m, n]

This example is CPU-focused and uses float32 inputs.
"""

import torch
import helion
import helion.language as hl
from helion.mlir import generate_mlir


@helion.kernel(static_shapes=True)
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Sum a 2D tensor along its last dimension."""
    m, _n = x.shape
    out = torch.empty((m,), dtype=torch.float32, device=x.device)

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)

    return out


def main() -> None:
    """Generate and print MLIR for the row-wise sum kernel."""
    device = torch.device("cpu")
    m, n = 128, 96
    x = torch.randn(m, n, dtype=torch.float32, device=device)

    print("=" * 80)
    print("Sum Reduction MLIR Generation (CPU)")
    print("=" * 80)
    print(f"Input shape: x={x.shape}")
    print(f"Output shape: ({m},)")

    module = generate_mlir(sum_kernel, [x])
    print("\nGenerated MLIR:")
    print("=" * 80)
    print(str(module))


if __name__ == "__main__":
    main()
