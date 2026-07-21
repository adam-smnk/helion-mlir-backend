"""
Element-wise Operations with MLIR Backend

Demonstrates several element-wise operations (add, multiply, ReLU) that generate
MLIR IR using linalg.generic for flexible element-wise computation.

This example shows:
- Simple element-wise addition
- Element-wise multiplication
- ReLU activation
- How MLIR handles multiple data types
"""

from __future__ import annotations

import helion
import helion.language as hl
from helion.mlir import generate_mlir
import torch


@helion.kernel(static_shapes=True)
def elementwise_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition: C = A + B

    Args:
        x, y: Input tensors of the same shape

    Returns:
        Output tensor with element-wise sum
    """
    m, n = x.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

    return out


@helion.kernel(static_shapes=True)
def elementwise_mul(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiplication: C = A * scale

    Args:
        x: Input tensor
        scale: Scalar tensor for scaling

    Returns:
        Output tensor with scaled values
    """
    m, n = x.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        # Broadcast scalar multiplication
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale

    return out


@helion.kernel(static_shapes=True)
def relu_activation(x: torch.Tensor) -> torch.Tensor:
    """
    ReLU activation: C = max(A, 0)

    Args:
        x: Input tensor

    Returns:
        Output tensor with ReLU applied
    """
    m, n = x.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])

    return out


def main() -> None:
    """Generate and display MLIR IR for element-wise operations."""

    device = torch.device("cpu")
    m, n = 32, 32

    # Test inputs
    x = torch.randn(m, n, dtype=torch.float32, device=device)
    y = torch.randn(m, n, dtype=torch.float32, device=device)
    scale = torch.tensor(2.5, dtype=torch.float32, device=device)

    print("=" * 80)
    print("Element-wise Operations MLIR Generation Examples")
    print("=" * 80)

    # Example 1: Addition
    print("\n" + "-" * 80)
    print("1. Element-wise Addition (x + y)")
    print("-" * 80)
    print(f"Input shapes: x={x.shape}, y={y.shape}")

    module = generate_mlir(elementwise_add, [x, y])
    ir_str = str(module)

    print(f"Generated IR snippet: {len(ir_str)} characters")
    print(
        f"Contains: linalg.generic={('linalg.generic' in ir_str)}, arith operations={('arith' in ir_str)}"
    )

    # Example 2: Multiplication with scalar
    print("\n" + "-" * 80)
    print("2. Element-wise Multiplication (x * scale)")
    print("-" * 80)
    print(f"Input shapes: x={x.shape}, scale={scale.shape}")

    module = generate_mlir(elementwise_mul, [x, scale])
    ir_str = str(module)
    print(f"Generated IR snippet: {len(ir_str)} characters")
    print(f"Contains: linalg.generic={('linalg.generic' in ir_str)}")

    # Example 3: ReLU
    print("\n" + "-" * 80)
    print("3. ReLU Activation (max(x, 0))")
    print("-" * 80)
    print(f"Input shape: x={x.shape}")

    module = generate_mlir(relu_activation, [x])
    ir_str = str(module)
    print(f"Generated IR snippet: {len(ir_str)} characters")
    print(
        f"Contains: linalg.generic={('linalg.generic' in ir_str)}, scf.forall={('scf.forall' in ir_str)}"
    )

    print("\n" + "=" * 80)
    print("Key observations:")
    print("=" * 80)
    print("- All operations use scf.forall for parallelization over tiles")
    print("- Element-wise ops use linalg.generic with custom compute blocks")
    print("- Broadcasts are handled implicitly by MLIR dialect semantics")
    print("- Tensor extraction and insertion for tile slices")


if __name__ == "__main__":
    main()
