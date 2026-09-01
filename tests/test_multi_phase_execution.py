"""End-to-end tests for the multi-phase (hl.barrier()) and host-tensor
interop driver -- the direct `backend="mlir"` call path only (see
bound_kernel.py::mlir_compile_config / _build_multi_phase_driver).

generate_mlir()/execute_mlir() intentionally do not support these kernels
(see test_multi_phase_generate_mlir_raises_clear_diagnostic below); they
require the real host-side driver built from host_prefix.py + phase_plan.py.
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401 registers "mlir" backend


def test_two_phase_barrier_kernel_direct_call():
    @helion.kernel(static_shapes=True, backend="mlir")
    def two_phase_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = two_phase_kernel(x, y)
    torch.testing.assert_close(result, (x + y) * 2.0)


def test_three_phase_chained_dependency_direct_call():
    @helion.kernel(static_shapes=True, backend="mlir")
    def three_phase_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        a = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        b = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            a[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            b[tm, tn] = a[tm, tn] * 2.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = b[tm, tn] - x[tm, tn]
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = three_phase_kernel(x, y)
    torch.testing.assert_close(result, ((x + y) * 2.0) - x)


def test_host_tensor_interop_direct_call():
    @helion.kernel(static_shapes=True, backend="mlir")
    def host_side_scale(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        scale = x.mean() * 100.0
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            s = hl.load(scale, [])
            out[tm, tn] = x[tm, tn] + s
        return out

    x = torch.randn(8, 8)
    result = host_side_scale(x)
    torch.testing.assert_close(result, x + x.mean() * 100.0)


def test_multi_phase_generate_mlir_raises_clear_diagnostic():
    from helion_mlir_backend import generate_mlir
    from helion_mlir_backend._compiler.mlir.support.errors import (
        UnsupportedOperationError,
    )

    @helion.kernel(static_shapes=True)
    def two_phase_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    with pytest.raises(UnsupportedOperationError, match="multi-phase"):
        generate_mlir(two_phase_kernel, [x, y])
