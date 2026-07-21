"""
GEGLU Kernel with MLIR Backend (CPU)

This example intentionally tracks the original `_geglu` math as closely as
possible:

     GELU(a) = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
     GEGLU(a, b) = GELU(a) * b

Notes on current backend limits:
1. The MLIR bridge is strongest on ATen ops; non-ATen composite call targets
    (for example internal `_gelu_tanh_approx`) may not lower yet.
2. Dtype/cast-heavy paths have improved in this branch, including
    tensor-attribute dtype casts in common GEGLU tracing patterns.
3. View/alias host tensors also have improved handling in this branch.
4. Flatten/view aliases in 1-D tiling forms still have verify-time edge cases
    in this backend path; this example intentionally uses N-D tiling.
5. If this example fails in your local environment, keep the polynomial form
    and simplify incrementally around casts to isolate unsupported nodes.
"""

import torch
import helion
import helion.language as hl
from helion.mlir import generate_mlir


@helion.kernel(static_shapes=True)
def geglu_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute GEGLU using explicit tanh-polynomial GELU approximation."""
    assert a.shape == b.shape

    out = torch.empty_like(a, dtype=torch.float32)

    sqrt_2_over_pi = 0.7978845608028654

    for tile in hl.tile(a.size()):
        a_vals = a[tile].to(torch.float32)
        b_vals = b[tile].to(torch.float32)

        a_cubed = a_vals * a_vals * a_vals
        tanh_arg = sqrt_2_over_pi * (a_vals + 0.044715 * a_cubed)
        tanh_result = torch.tanh(tanh_arg)
        gelu_a = 0.5 * a_vals * (1.0 + tanh_result)

        out[tile] = gelu_a * b_vals

    return out


def main() -> None:
    """Generate and print MLIR for the GEGLU kernel."""
    device = torch.device("cpu")
    shape = (8, 64, 128)

    a = torch.randn(shape, dtype=torch.float32, device=device)
    b = torch.randn(shape, dtype=torch.float32, device=device)

    print("=" * 80)
    print("GEGLU MLIR Generation (CPU)")
    print("=" * 80)
    print(f"Input shape: {shape}")

    module = generate_mlir(geglu_kernel, [a, b])
    print("\nGenerated MLIR:")
    print("=" * 80)
    print(str(module))


if __name__ == "__main__":
    main()
