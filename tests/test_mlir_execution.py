"""Tests for MLIR kernel execution via lighthouse (CPU).

Covers both execution paths:
- ``execute_mlir``  – explicit generate_mlir + MLIRBackend.execute_mlir
- ``direct call``   – @helion.kernel(backend="mlir") called like a normal Python function
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from helion_mlir_backend import generate_mlir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend():
    from helion._compiler.backend_registry import get_backend_class

    return get_backend_class("mlir")()


def _allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ab_32x32():
    torch.manual_seed(0)
    return torch.randn(32, 32), torch.randn(32, 32)


@pytest.fixture
def abc_16x64():
    torch.manual_seed(1)
    return torch.randn(16, 64), torch.randn(16, 64), torch.randn(16, 64)


# ---------------------------------------------------------------------------
# execute_mlir path
# ---------------------------------------------------------------------------


class TestExecuteMlir:
    """Tests for MLIRBackend.execute_mlir (explicit two-step API)."""

    def test_add_execute_mlir(self, ab_32x32):
        """Elementwise add via execute_mlir matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_kernel, [A, B])
        C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
        assert _allclose(C, A + B)

    def test_mul_execute_mlir(self, ab_32x32):
        """Elementwise multiply via execute_mlir matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True)
        def mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(mul_kernel, [A, B])
        C = _backend().execute_mlir(mlir_module, A, B, kernel_name="mul_kernel")
        assert _allclose(C, A * B)

    def test_scale_add_execute_mlir(self, abc_16x64):
        """Fused scale-add (x * 2 + y) via execute_mlir."""
        X, Y, _ = abc_16x64

        @helion.kernel(static_shapes=True)
        def scale_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0 + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(scale_add_kernel, [X, Y])
        C = _backend().execute_mlir(mlir_module, X, Y, kernel_name="scale_add_kernel")
        assert _allclose(C, X * 2.0 + Y)

    def test_output_shape_execute_mlir(self, ab_32x32):
        """Output tensor has correct shape and dtype."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_kernel, [A, B])
        C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
        assert C.shape == A.shape
        assert C.dtype == torch.float32

    def test_cpu_only_guard(self, ab_32x32):
        """execute_mlir raises NotImplementedError for non-CPU tensors."""
        A, B = ab_32x32
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_kernel, [A, B])
        with pytest.raises(NotImplementedError, match="CPU"):
            _backend().execute_mlir(
                mlir_module, A.cuda(), B.cuda(), kernel_name="add_kernel"
            )


# ---------------------------------------------------------------------------
# Direct-call path (backend="mlir" integrated into helion runtime)
# ---------------------------------------------------------------------------


class TestDirectCall:
    """Tests for @helion.kernel(backend="mlir") called directly."""

    def test_add_direct(self, ab_32x32):
        """Direct add kernel call matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True, backend="mlir")
        def add_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        assert _allclose(add_direct(A, B), A + B)

    def test_mul_direct(self, ab_32x32):
        """Direct multiply kernel call matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True, backend="mlir")
        def mul_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * y[tile_m, tile_n]
            return out

        assert _allclose(mul_direct(A, B), A * B)

    def test_three_input_direct(self, abc_16x64):
        """Direct call with three tensor inputs."""
        X, Y, Z = abc_16x64

        @helion.kernel(static_shapes=True, backend="mlir")
        def add3_direct(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = (
                    x[tile_m, tile_n] + y[tile_m, tile_n] + z[tile_m, tile_n]
                )
            return out

        assert _allclose(add3_direct(X, Y, Z), X + Y + Z)

    def test_result_is_cached(self, ab_32x32):
        """Second call reuses the compiled JIT function (no recompilation)."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True, backend="mlir")
        def add_cached(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        C1 = add_cached(A, B)
        C2 = add_cached(A, B)
        assert _allclose(C1, C2)

    def test_direct_matches_execute_mlir(self, ab_32x32):
        """Direct call and execute_mlir produce identical results."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True, backend="mlir")
        def add_both(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_both, [A, B])
        C_explicit = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_both")
        C_direct = add_both(A, B)
        assert torch.equal(C_explicit, C_direct)
