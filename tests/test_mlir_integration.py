"""Integration tests for MLIR backend with downstream compilation targets.

Tests that the generated MLIR is valid and suitable for downstream
compilation pipelines (e.g., Triton, LLVM).
"""

import pytest
import torch
import helion
import helion.language as hl
from helion_mlir_backend import generate_mlir


class TestDownstreamIntegration:
    """Test integration with downstream compilers."""

    def test_matmul_mlir_structure(self):
        """Test that matmul MLIR has expected structure for Triton."""
        @helion.kernel(static_shapes=True)
        def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        device = torch.device("cpu")
        x = torch.randn(128, 256, dtype=torch.float32, device=device)
        y = torch.randn(256, 192, dtype=torch.float32, device=device)

        module = generate_mlir(matmul_kernel, [x, y])
        ir_str = str(module)

        # Verify key MLIR constructs for downstream
        assert "func.func" in ir_str, "Should have func dialect"
        assert "scf.forall" in ir_str, "Should have parallel loops"
        assert "scf.for" in ir_str, "Should have sequential loops"
        assert "linalg.matmul" in ir_str, "Should have linalg operations"
        assert "tensor." in ir_str, "Should use tensor dialect"
        assert "f32" in ir_str, "Should have float32 type"

    def test_mlir_module_validity(self):
        """Test that generated MLIR is a valid module."""
        @helion.kernel(static_shapes=True)
        def simple_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(simple_kernel, [x])

        # Module should be properly formed
        assert module is not None
        assert hasattr(module, "body"), "Module should have body"
        ir_str = str(module)
        assert "module" in ir_str.lower(), "Should contain module declaration"

    def test_elementwise_ops_linalg_generic(self):
        """Test that elementwise ops use linalg.generic."""
        @helion.kernel(static_shapes=True)
        def elementwise_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(elementwise_kernel, [x, y])
        ir_str = str(module)

        # Elementwise operations should use linalg.generic
        assert "linalg.generic" in ir_str or "arith." in ir_str, \
            "Should use linalg.generic or arith ops for elementwise"

    def test_different_dtypes_in_module(self):
        """Test that multiple dtypes are properly handled."""
        @helion.kernel(static_shapes=True)
        def mixed_dtype_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            # Work with the input dtype
            out = torch.zeros((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        
        # Test with float64
        x = torch.randn(32, 64, dtype=torch.float64, device=device)
        module = generate_mlir(mixed_dtype_kernel, [x])
        ir_str = str(module)
        assert "f64" in ir_str, "Should preserve float64 dtype"

    def test_non_square_matrices(self):
        """Test MLIR generation with non-square matrix dimensions."""
        @helion.kernel(static_shapes=True)
        def tall_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape  # Tall matrix
            k2, n = y.shape  # Wide matrix
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        device = torch.device("cpu")
        x = torch.randn(256, 128, dtype=torch.float32, device=device)  # Tall
        y = torch.randn(128, 64, dtype=torch.float32, device=device)   # Wide

        module = generate_mlir(tall_matrix, [x, y])
        ir_str = str(module)

        # Should handle various dimensions
        assert "256" in ir_str, "Should reflect tall dimension"
        assert "scf.forall" in ir_str, "Should have parallel iteration"

    def test_fused_operations(self):
        """Test MLIR generation for fused operations."""
        @helion.kernel(static_shapes=True)
        def fused_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                # Apply relu to the result
                out[tile_m, tile_n] = torch.relu(acc)
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(128, 96, dtype=torch.float32, device=device)

        module = generate_mlir(fused_kernel, [x, y])
        ir_str = str(module)

        # Should have both matmul and relu operations
        assert "linalg.matmul" in ir_str or "linalg.generic" in ir_str
        # Verify function structure is preserved
        assert "func.return" in ir_str or "return" in ir_str

    def test_multiple_tiles(self):
        """Test MLIR with multiple tiling levels."""
        @helion.kernel(static_shapes=True)
        def multi_tile_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                # Could further subdivide within tiles
                result = torch.relu(x[tile_m, tile_n])
                out[tile_m, tile_n] = result
            return out

        device = torch.device("cpu")
        x = torch.randn(256, 256, dtype=torch.float32, device=device)

        module = generate_mlir(multi_tile_kernel, [x])
        ir_str = str(module)

        # Should have structured tiling
        assert "scf.forall" in ir_str
        assert "256" in ir_str

    def test_mlir_ir_printable(self):
        """Test that MLIR can be printed without errors."""
        @helion.kernel(static_shapes=True)
        def printable_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(32, 64, dtype=torch.float32, device=device)
        y = torch.randn(32, 64, dtype=torch.float32, device=device)

        module = generate_mlir(printable_kernel, [x, y])
        ir_str = str(module)

        # Should be printable and non-empty
        assert len(ir_str) > 0
        assert "func.func" in ir_str

    def test_module_has_proper_function_signature(self):
        """Test that generated module has proper function signature."""
        @helion.kernel(static_shapes=True)
        def signature_test(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        device = torch.device("cpu")
        x = torch.randn(64, 128, dtype=torch.float32, device=device)
        y = torch.randn(64, 128, dtype=torch.float32, device=device)

        module = generate_mlir(signature_test, [x, y])
        ir_str = str(module)

        # Function should have proper signature
        assert "func.func" in ir_str
        assert "tensor<" in ir_str, "Should have tensor types"
        assert "x32" in ir_str or "f32" in ir_str, "Should have float type"
        assert "func.return" in ir_str or "return" in ir_str, "Should have return statement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
