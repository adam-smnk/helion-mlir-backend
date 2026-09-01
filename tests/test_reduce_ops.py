"""Numerical tests for reduction & broadcast ATen ops routed through the
generic torch-mlir helper path (Phase 5 of the MLIR backend cleanup)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from helion_mlir_backend import generate_mlir

try:
    from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

    _backend = MLIRBackend
except Exception:  # pragma: no cover
    _backend = None


def _execute(module, *tensors, kernel_name):
    return _backend().execute_mlir(module, *tensors, kernel_name=kernel_name)


def test_sum_reduction():
    @helion.kernel(static_shapes=True)
    def sum_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].sum(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(sum_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="sum_kernel")
    torch.testing.assert_close(actual, x.sum(dim=1))


def test_min_reduction():
    @helion.kernel(static_shapes=True)
    def min_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].amin(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(min_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="min_kernel")
    torch.testing.assert_close(actual, x.amin(dim=1))


def test_max_reduction():
    @helion.kernel(static_shapes=True)
    def max_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=x.dtype, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].amax(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(max_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="max_kernel")
    torch.testing.assert_close(actual, x.amax(dim=1))


def test_argmax_reduction():
    @helion.kernel(static_shapes=True)
    def argmax_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m,), dtype=torch.int64, device=x.device)
        for tm in hl.tile(m):
            out[tm] = x[tm, :].argmax(dim=1)
        return out

    x = torch.randn(32, 16)
    module = generate_mlir(argmax_kernel, [x], config=helion.Config(block_sizes=[8]))
    actual = _execute(module, x, kernel_name="argmax_kernel")
    torch.testing.assert_close(actual, x.argmax(dim=1))


def test_rank_mismatch_broadcast_add():
    """A 1-D bias added to a 2-D accumulator (rank-expanding broadcast).

    Regression test for a real bug found while auditing examples/: the ATen
    helper pre-build path used to normalize all operands of a broadcasting op
    to a common *rank-expanded* shape purely for helper-signature purposes,
    without ever materializing that expansion in the actual call-site MLIR
    value -- causing every rank-mismatched broadcast (e.g. matmul + 1-D bias)
    to fail with a helper signature mismatch. Fixed by restricting that
    same-shape normalization to same-rank operands only.
    """

    @helion.kernel(static_shapes=True)
    def matmul_add(
        x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc + bias[tile_n]
        return out

    x = torch.randn(64, 128)
    y = torch.randn(128, 96)
    bias = torch.randn(96)
    module = generate_mlir(matmul_add, [x, y, bias])
    actual = _execute(module, x, y, bias, kernel_name="matmul_add")
    torch.testing.assert_close(actual, x @ y + bias, atol=1e-3, rtol=1e-3)


def test_multi_output_kernel_raises_clear_diagnostic():
    """A kernel returning two tensors must fail with a clear, actionable

    diagnostic, not the previous confusing deep error.

    Regression test for a real bug found via Phase H root-cause analysis:
    `OutputTensorResolver.resolve()`'s multi-output guard only inspected each
    device-IR loop-body FX graph's own ``output`` node, which stays trivial
    (``None``) for this pattern -- outputs are mutated host tensors written
    via ``store``, not returned as graph results -- so the guard never fired.
    Execution instead silently fell through to `tensor_params[-1]`, an
    arbitrary (and wrong) tensor pick, surfacing as a baffling downstream
    `ModuleBuilderError: No value for tensor node <input name>`. Fixed by
    also detecting 2+ distinct non-input store targets (the shape this
    pattern actually takes) and raising `UnsupportedOperationError` there.
    """
    from helion_mlir_backend._compiler.mlir.support.errors import (
        UnsupportedOperationError,
    )

    @helion.kernel(static_shapes=True)
    def two_outputs(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = x[tm, tn] + y[tm, tn]
            out2[tm, tn] = x[tm, tn] - y[tm, tn]
        return out1, out2

    x = torch.randn(16, 16)
    y = torch.randn(16, 16)
    with pytest.raises(UnsupportedOperationError, match="multi-output kernel"):
        generate_mlir(two_outputs, [x, y])
