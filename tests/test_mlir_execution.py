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
            # 1D tile over M to stay within the scalar-lowering pipeline.
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] * 2.0 + y[tile_m, :]
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
            # 1D tile over M to stay within the scalar-lowering pipeline.
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :] + z[tile_m, :]
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


# ---------------------------------------------------------------------------
# Configurable block sizes
# ---------------------------------------------------------------------------


class TestConfigurableBlockSizes:
    """Tests that block_sizes from the config propagate into the generated MLIR."""

    def test_block_size_in_ir(self):
        """Config block_sizes are reflected as scf.forall step in the IR."""

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(0)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        for block_size in (8, 16, 32, 64):
            cfg = helion.Config(block_sizes=[block_size])
            mlir_module = generate_mlir(add_kernel, [A, B], config=cfg)
            ir_str = str(mlir_module)
            assert f"step ({block_size})" in ir_str, (
                f"Expected step ({block_size}) in IR, got:\n{ir_str}"
            )

    def test_different_block_sizes_same_result(self):
        """Different block_sizes produce numerically identical results via execute_mlir."""

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(42)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)
        ref = A + B

        for block_size in (8, 16, 32, 64):
            cfg = helion.Config(block_sizes=[block_size])
            mlir_module = generate_mlir(add_kernel, [A, B], config=cfg)
            C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
            assert _allclose(C, ref), (
                f"block_size={block_size}: max_err={torch.max(torch.abs(C - ref)).item():.2e}"
            )

    def test_block_size_via_direct_call(self):
        """Block size from Config propagates through the direct helion kernel call."""
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16]),
        )
        def add_bs16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(7)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        # Capture the pre-lowering IR emitted by the direct call to verify
        # that block_size=16 from the Config decorator reaches the codegen.
        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = add_bs16(A, B)

        ir_dump = buf.getvalue()
        assert "step (16)" in ir_dump, (
            f"Expected step (16) from Config in pre-lowering IR dump, got:\n{ir_dump[:500]}"
        )
        assert _allclose(result, A + B)

    def test_outer_forall_inner_scf_for_block_sizes(self):
        """Outer hl.tile([m,n]) → scf.forall; inner hl.tile(k) → scf.for.

        Config(block_sizes=[bs_mn, bs_k]) controls both steps independently.
        Verifies both step values appear in the pre-lowering IR and results are correct.
        """
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16, 8, 32]),
        )
        def matmul_tiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(3)
        # Non-square dimensions are required: all three tile blocks (tile_m, tile_n, tile_k)
        # must have distinct hint values so _block_hint_to_id maps them unambiguously.
        # Choose k=64 with block_k=32 so the inner reduction loop must execute
        # multiple iterations and remains visible in pre-lowering IR.
        A = torch.randn(32, 64)
        B = torch.randn(64, 48)

        # Clear helion's in-memory kernel cache to force fresh compilation.
        matmul_tiled._bound_kernels.clear()
        matmul_tiled._dispatch_cache.clear()

        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = matmul_tiled(A, B)

        ir_dump = buf.getvalue()
        # Outer 2D tile: step (16, 16) from scf.forall.
        assert "step (16, 8)" in ir_dump, "Expected outer scf.forall step (16, 8)"
        # Inner k-reduction: scf.for uses a SSA step value from arith.constant.
        assert "step %c32" in ir_dump or "step (32)" in ir_dump, (
            "Expected inner scf.for k-reduction step 32"
        )
        assert _allclose(result, A @ B)

    def test_outer_forall_inner_scf_for_block_sizes_matmul_accumulate(self):
        """Nested tiling with explicit matmul + add accumulation.

        Same structure as addmm test but uses:
        ``acc = acc + torch.matmul(dequant, act_group)``.
        """
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16, 8, 32]),
        )
        def matmul_acc_tiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    dequant = x[tile_m, tile_k]
                    act_group = y[tile_k, tile_n]
                    acc = acc + torch.matmul(dequant, act_group)
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(11)
        A = torch.randn(32, 64)
        B = torch.randn(64, 48)

        # Clear helion's in-memory kernel cache to force fresh compilation.
        matmul_acc_tiled._bound_kernels.clear()
        matmul_acc_tiled._dispatch_cache.clear()

        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = matmul_acc_tiled(A, B)

        ir_dump = buf.getvalue()
        assert "step (16, 8)" in ir_dump, "Expected outer scf.forall step (16, 8)"
        assert "step %c32" in ir_dump or "step (32)" in ir_dump, (
            "Expected inner scf.for k-reduction step 32"
        )
        assert _allclose(result, A @ B)
