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

    def test_execute_mlir_rejects_empty_inputs(self):
        with pytest.raises(TypeError, match="non-empty"):
            _backend().execute_mlir(None)

    def test_execute_mlir_rejects_non_tensor_inputs(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _backend().execute_mlir(None, torch.ones(2), 1)

    def test_execute_mlir_rejects_mixed_devices(self):
        cpu_tensor = torch.ones(2)
        meta_tensor = torch.ones(2, device="meta")
        with pytest.raises(ValueError, match="same device"):
            _backend().execute_mlir(None, cpu_tensor, meta_tensor)

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

    def test_matmul_full_rhs_slice_execute_mlir_regression(self):
        """Regression: full RHS slice in tiled matmul compiles and runs correctly.

        Mirrors the example pattern where B is loaded with full slices inside
        an outer tile loop and multiplied with a tiled slice of A.
        """

        @helion.kernel(static_shapes=True)
        def matmul_full_rhs_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            assert k == k2

            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                a_tile = hl.load(x, [tile_m, slice(None)])
                b_full = hl.load(y, [slice(None), slice(None)])
                c_tile = a_tile @ b_full
                hl.store(out, [tile_m, slice(None)], c_tile)
            return out

        torch.manual_seed(123)
        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        ref = a @ b

        mlir_module = generate_mlir(matmul_full_rhs_slice, [a, b])
        c = _backend().execute_mlir(
            mlir_module,
            a,
            b,
            kernel_name="matmul_full_rhs_slice",
        )

        assert torch.isfinite(c).all()
        assert _allclose(c, ref)

    def test_concat2d_dim1_simple_generate_mlir_regression(self):
        """Regression: concat2d_dim1_simple pattern lowers and emits kernel symbol."""

        @helion.kernel(static_shapes=True)
        def concat2d_dim1_simple_kernel(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            assert x.size(0) == y.size(0)
            out = torch.empty(
                [x.size(0), x.size(1) + y.size(1)],
                dtype=x.dtype,
                device=x.device,
            )
            n1 = x.size(1)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :n1] = x[tile_m, :]
                out[tile_m, n1:] = y[tile_m, :]
            return out

        torch.manual_seed(29)
        x = torch.randn(64, 40)
        y = torch.randn(64, 24)

        mlir_module = generate_mlir(concat2d_dim1_simple_kernel, [x, y])
        assert "concat2d_dim1_simple_kernel" in str(mlir_module)
        # TODO(helion-mlir): Runtime broken

    def test_broadcast_matmul_generate_mlir_regression(self):
        """Regression: broadcasted batch matmul kernel lowers and emits the kernel symbol."""

        @helion.kernel(static_shapes=True)
        def broadcast_matmul_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = w.size()
            assert k == k2

            x_2d = x.reshape([b * m, k])
            out_2d = torch.empty(
                [b * m, n],
                device=x.device,
                dtype=torch.promote_types(x.dtype, w.dtype),
            )
            for tile_bm, tile_n in hl.tile([b * m, n]):
                acc = hl.zeros([tile_bm, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x_2d[tile_bm, tile_k], w[tile_k, tile_n])
                out_2d[tile_bm, tile_n] = acc

            return out_2d.view(b, m, n)

        torch.manual_seed(41)
        x = torch.randn(4, 8, 16)
        w = torch.randn(16, 12)

        mlir_module = generate_mlir(broadcast_matmul_kernel, [x, w])
        assert "broadcast_matmul_kernel" in str(mlir_module)
        # TODO(helion-mlir): Runtime broken

    def test_flat_gather_row_sum_indexing_compile(self):
        """Regression: flattened gather + reduction with standard hl.tile matches reference."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[32, 16]),
        )
        def flat_gather_row_sum_kernel(
            x_data: torch.Tensor,
            flat_idx: torch.Tensor,
        ) -> torch.Tensor:
            num_rows, m = x_data.shape
            x_flat = x_data.view(-1)
            out = torch.zeros((num_rows,), dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(num_rows):
                idx_tile = hl.load(flat_idx, [tile_b, slice(None)])
                x_slice = hl.load(x_flat, [idx_tile])
                out[tile_b] = x_slice.sum(dim=1)
            return out

        torch.manual_seed(7)
        x_data = torch.randn(64, 16)
        row_ids = torch.arange(x_data.shape[0], dtype=torch.int64)[:, None]
        col_ids = torch.arange(x_data.shape[1], dtype=torch.int64)[None, :]
        flat_idx = row_ids * x_data.shape[1] + col_ids
        expected = x_data.sum(dim=1)

        module = generate_mlir(flat_gather_row_sum_kernel, [x_data, flat_idx])
        actual = _backend().execute_mlir(
            module,
            x_data,
            flat_idx,
            kernel_name="flat_gather_row_sum_kernel",
        )

        assert _allclose(actual, expected)

    def test_flat_gather_2d_copy_indexing_compile(self):
        """Regression: flattened 2D gather copy with standard hl.tile preserves values."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[32, 16]),
        )
        def flat_gather_2d_copy_kernel(
            x_data: torch.Tensor,
            flat_idx: torch.Tensor,
        ) -> torch.Tensor:
            num_rows, m = x_data.shape
            x_flat = x_data.view(-1)
            out = torch.empty((num_rows, m), dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(num_rows):
                idx_tile = hl.load(flat_idx, [tile_b, slice(None)])
                out[tile_b, :] = hl.load(x_flat, [idx_tile])
            return out

        torch.manual_seed(11)
        x_data = torch.randn(64, 16)
        row_ids = torch.arange(x_data.shape[0], dtype=torch.int64)[:, None]
        col_ids = torch.arange(x_data.shape[1], dtype=torch.int64)[None, :]
        flat_idx = row_ids * x_data.shape[1] + col_ids

        module = generate_mlir(flat_gather_2d_copy_kernel, [x_data, flat_idx])
        actual = _backend().execute_mlir(
            module,
            x_data,
            flat_idx,
            kernel_name="flat_gather_2d_copy_kernel",
        )

        assert _allclose(actual, x_data)

    def test_tile_index_usage_execute_mlir(self):
        """tile.index can be used for indexed load/store in a 1D tiled loop."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[16]),
        )
        def tile_index_kernel(x: torch.Tensor) -> torch.Tensor:
            n = x.shape[0]
            out = torch.empty((n,), dtype=x.dtype, device=x.device)

            for tile_n in hl.tile(n):
                out[tile_n] = hl.load(x, [tile_n.index]) + 1.0

            return out

        torch.manual_seed(23)
        x = torch.randn(64)
        expected = x + 1.0

        module = generate_mlir(tile_index_kernel, [x])
        actual = _backend().execute_mlir(
            module,
            x,
            kernel_name="tile_index_kernel",
        )

        assert _allclose(actual, expected)

    def test_grid_batched_matmul_execute_mlir(self):
        """`hl.grid` scalar indices produce numerically correct batched matmul."""

        @helion.kernel(static_shapes=True)
        def grid_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nb, m, k = a.shape
            out = torch.zeros((nb, m, b.shape[2]), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                out[i, :, :] = torch.matmul(a[i, :, :], b[i, :, :])
            return out

        torch.manual_seed(7)
        a = torch.randn(4, 32, 32)
        b = torch.randn(4, 32, 32)

        module = generate_mlir(grid_bmm, [a, b])
        actual = _backend().execute_mlir(module, a, b, kernel_name="grid_bmm")

        assert _allclose(actual, torch.bmm(a, b))

    def test_block_packing_kernels_execute_mlir(self):
        """Nested tiled A/B panel packing matches contiguous PyTorch references."""
        from helion_block_pack import pack_a
        from helion_block_pack import pack_b

        torch.manual_seed(41)
        a = torch.randn(64, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 64, dtype=torch.bfloat16)

        packed_a = pack_a(a, 32)
        packed_b = pack_b(b, 32)

        torch.testing.assert_close(
            packed_a,
            a.view(64, 2, 32).permute(1, 0, 2).contiguous(),
        )
        torch.testing.assert_close(
            packed_b,
            b.view(64, 2, 32).permute(1, 0, 2).contiguous(),
        )

    def test_unpack_grid_tile_reordered_store_execute_mlir(self):
        """Store index order can differ from loop declaration order.

        ``out[tm, panel, :]`` writes the tile index before the grid index even
        though the grid loop is declared first; the grid block id must still
        land on output dimension 1 (not 0).
        """

        @helion.kernel(static_shapes=True)
        def unpack_panels(src: torch.Tensor) -> torch.Tensor:
            n_panels, m, bn = src.shape
            out = torch.empty((m, n_panels, bn), dtype=src.dtype, device=src.device)
            for panel in hl.grid(n_panels):
                for tm in hl.tile(m):
                    out[tm, panel, :] = src[panel, tm, :]
            return out

        torch.manual_seed(37)
        src = torch.randn(3, 8, 8)
        module = generate_mlir(
            unpack_panels,
            [src],
            config=helion.Config(block_sizes=[1, 4]),
        )
        actual = _backend().execute_mlir(module, src, kernel_name="unpack_panels")

        assert _allclose(actual, src.permute(1, 0, 2).contiguous())

    @pytest.mark.parametrize("size", [64, 128])
    def test_packed_rhs_matmul_execute_mlir(self, size):
        """A packed RHS remains numerically correct through tiled matmul consumption."""
        from helion_block_packed_f32_repro import matmul_packed_b_f32
        from helion_block_packed_f32_repro import pack_b_panels_f32
        from helion_block_packed_f32_repro import unpack_panel_major

        torch.manual_seed(43)
        block_n = 32
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        b3 = b.view(size, size // block_n, block_n).contiguous()

        packed_b = pack_b_panels_f32(b3)
        panel_result = matmul_packed_b_f32(a, packed_b)
        actual = unpack_panel_major(panel_result)

        assert _allclose(actual, torch.mm(a, b))

    def test_nested_grid_copy_execute_mlir(self):
        """Nested unit-step grid indices preserve both outer dimensions."""

        @helion.kernel(static_shapes=True)
        def nested_grid_copy(a: torch.Tensor) -> torch.Tensor:
            nb, nc, m = a.shape
            out = torch.zeros_like(a)
            for i in hl.grid(nb):
                for j in hl.grid(nc):
                    out[i, j, :] = a[i, j, :] + 1.0
            return out

        torch.manual_seed(31)
        a = torch.randn(2, 3, 8)
        module = generate_mlir(
            nested_grid_copy,
            [a],
            config=helion.Config(block_sizes=[1, 1, 8]),
        )
        actual = _backend().execute_mlir(module, a, kernel_name="nested_grid_copy")

        assert _allclose(actual, a + 1.0)

    @pytest.mark.parametrize("shape", [(2, 3, 8), (3, 2, 10)])
    def test_grid_tile_slice_execute_mlir(self, shape):
        """Scalar grid plus exact/ragged trailing tile slices stay numerical."""

        @helion.kernel(static_shapes=True)
        def grid_tile_copy(a: torch.Tensor) -> torch.Tensor:
            nb, nc, m = a.shape
            out = torch.zeros_like(a)
            for i in hl.grid(nb):
                for tm in hl.tile(m):
                    out[i, :, tm] = a[i, :, tm] * 2.0 + 1.0
            return out

        torch.manual_seed(sum(shape))
        a = torch.randn(*shape)
        module = generate_mlir(
            grid_tile_copy,
            [a],
            config=helion.Config(block_sizes=[1, 1, 4]),
        )
        actual = _backend().execute_mlir(module, a, kernel_name="grid_tile_copy")

        assert _allclose(actual, a * 2.0 + 1.0)

    def test_tile_begin_execute_mlir(self):
        """`tile.begin` contributes the correct per-tile offset value."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[16]))
        def tile_begin_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.begin
            return out

        torch.manual_seed(11)
        x = torch.randn(64, 32)

        module = generate_mlir(
            tile_begin_kernel, [x], config=helion.Config(block_sizes=[16])
        )
        actual = _backend().execute_mlir(module, x, kernel_name="tile_begin_kernel")

        offsets = (torch.arange(64) // 16 * 16).to(torch.float32).unsqueeze(1)
        assert _allclose(actual, x + offsets)

    @pytest.mark.parametrize(
        "block_sizes", [[16, 32, 64], [32, 32, 32], [16, 16, 16], [32, 64, 32]]
    )
    def test_nested_tile_block_sizes_execute_mlir(self, block_sizes):
        """Repeated nested block sizes still compute a correct matmul."""

        @helion.kernel(static_shapes=True)
        def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc
            return out

        torch.manual_seed(3)
        x = torch.randn(128, 128)
        y = torch.randn(128, 128)

        module = generate_mlir(
            mm, [x, y], config=helion.Config(block_sizes=block_sizes)
        )
        actual = _backend().execute_mlir(module, x, y, kernel_name="mm")

        assert _allclose(actual, x @ y)

    def test_transposed_rhs_matmul_execute_mlir(self):
        """A transposed RHS lowered via linalg.contract is numerically correct."""

        @helion.kernel(static_shapes=True)
        def mm_transposed_b(x: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            n, k2 = yt.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], yt[tn, tk].permute(1, 0))
                out[tm, tn] = acc
            return out

        torch.manual_seed(5)
        x = torch.randn(64, 64)
        yt = torch.randn(64, 64)

        module = generate_mlir(
            mm_transposed_b, [x, yt], config=helion.Config(block_sizes=[16, 32, 64])
        )
        actual = _backend().execute_mlir(module, x, yt, kernel_name="mm_transposed_b")

        assert _allclose(actual, x @ yt.t())

    def test_dtype_cast_epilogue_execute_mlir(self):
        """An explicit `.to(bfloat16)` epilogue produces a bf16 result."""

        @helion.kernel(static_shapes=True)
        def cast_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tm, tn] = (x[tm, tn] * 2.0).to(torch.bfloat16)
            return out

        torch.manual_seed(13)
        x = torch.randn(64, 64)

        module = generate_mlir(cast_kernel, [x])
        actual = _backend().execute_mlir(module, x, kernel_name="cast_kernel")

        assert actual.dtype == torch.bfloat16
        assert _allclose(actual, (x * 2.0).to(torch.bfloat16))

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

    def test_matmul_full_rhs_slice_direct_regression(self):
        """Regression: direct MLIR backend call handles full RHS slice matmul."""

        @helion.kernel(static_shapes=True, backend="mlir")
        def matmul_full_rhs_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            assert k == k2

            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                a_tile = hl.load(x, [tile_m, slice(None)])
                b_full = hl.load(y, [slice(None), slice(None)])
                c_tile = a_tile @ b_full
                hl.store(out, [tile_m, slice(None)], c_tile)
            return out

        torch.manual_seed(321)
        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        ref = a @ b

        c = matmul_full_rhs_slice(a, b)
        assert torch.isfinite(c).all()
        assert _allclose(c, ref)


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

    def test_decorator_config_is_used_without_kwarg(self):
        """A config on @helion.kernel applies even when generate_mlir omits it."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(0)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        ir_str = str(generate_mlir(add_kernel, [A, B]))

        assert "step (8)" in ir_str, (
            f"decorator block_sizes=[8] was ignored, got:\n{ir_str}"
        )

    def test_explicit_config_overrides_decorator_config(self):
        """An explicit config= argument wins over the decorator's config."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(0)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        ir_str = str(
            generate_mlir(add_kernel, [A, B], config=helion.Config(block_sizes=[32]))
        )

        assert "step (32)" in ir_str
        assert "step (8)" not in ir_str

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
        assert _allclose(result, A @ B)

    def test_outer_inner_loops_eltwise_block_sizes(self):
        """Nested outer/inner loops with eltwise ops honor configured steps."""
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16, 8]),
        )
        def add_nested(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        torch.manual_seed(19)
        A = torch.randn(64, 48)
        B = torch.randn(64, 48)

        # Clear helion's in-memory kernel cache to force fresh compilation.
        add_nested._bound_kernels.clear()
        add_nested._dispatch_cache.clear()

        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = add_nested(A, B)

        ir_dump = buf.getvalue()
        assert "step (16)" in ir_dump, "Expected outer scf.forall step (16)"
        assert "step %c8" in ir_dump or "step (8)" in ir_dump, (
            "Expected inner loop step 8"
        )

        assert _allclose(result, A + B)
