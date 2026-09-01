"""Numerical tests for reduction & broadcast ATen ops routed through the
generic torch-mlir helper path (Phase 5 of the MLIR backend cleanup)."""

from __future__ import annotations

import helion
import helion.language as hl
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
