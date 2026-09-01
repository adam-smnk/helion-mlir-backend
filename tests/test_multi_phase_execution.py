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


def test_host_tensor_computed_before_phases_and_used_in_later_phase():
    """Host work can precede phases and feed a later phase by ``hl.load``.

    Helion disallows ordinary statements between top-level device loops, so
    this is the valid syntax for host-side interop used by a later phase.
    """

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    def later_phase_uses_host_scale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        scale = x.mean() * 3.0
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            s = hl.load(scale, [])
            out[tm, tn] = mid[tm, tn] * s
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = later_phase_uses_host_scale(x, y)
    torch.testing.assert_close(result, (x + y) * (x.mean() * 3.0))


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


def test_two_phase_kernel_with_multi_output_final_phase():
    """Combines both features: a 2-phase kernel whose final phase returns
    two tensors (return out1, out2), exercising the N-output-per-phase
    machinery together with cross-phase threading in the same kernel."""

    @helion.kernel(static_shapes=True, backend="mlir")
    def two_phase_multi_output(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = mid[tm, tn] * 2.0
            out2[tm, tn] = mid[tm, tn] - x[tm, tn]
        return out1, out2

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = two_phase_multi_output(x, y)
    assert isinstance(result, list) and len(result) == 2
    mid = x + y
    torch.testing.assert_close(result[0], mid * 2.0)
    torch.testing.assert_close(result[1], mid - x)


def test_three_output_kernel_direct_call():
    """Multi-output stress test: 3 outputs from one hl.tile() loop, via the
    direct call path (which takes the unchanged single-phase fast path
    since there's no hl.barrier()/extra host tensor here)."""

    @helion.kernel(static_shapes=True, backend="mlir")
    def three_outputs(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out3 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = x[tm, tn] + y[tm, tn]
            out2[tm, tn] = x[tm, tn] - y[tm, tn]
            out3[tm, tn] = x[tm, tn] * y[tm, tn]
        return out1, out2, out3

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = three_outputs(x, y)
    assert isinstance(result, list) and len(result) == 3
    torch.testing.assert_close(result[0], x + y)
    torch.testing.assert_close(result[1], x - y)
    torch.testing.assert_close(result[2], x * y)


def test_four_phase_chained_dependency_direct_call():
    """Stress test beyond 2-3 phases: 4 phases, each depending on the last."""

    @helion.kernel(static_shapes=True, backend="mlir")
    def four_phase_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        a = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        b = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        c = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            a[tm, tn] = x[tm, tn] + 1.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            b[tm, tn] = a[tm, tn] * 2.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            c[tm, tn] = b[tm, tn] - 1.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = c[tm, tn] / 2.0
        return out

    x = torch.randn(8, 8)
    result = four_phase_kernel(x)
    expected = (((x + 1.0) * 2.0) - 1.0) / 2.0
    torch.testing.assert_close(result, expected)


def test_bad_return_statement_raises_clear_diagnostic():
    """A multi-phase kernel's final `return` must be a plain name or tuple
    of names; an arbitrary expression must raise a clear diagnostic instead
    of silently returning the wrong (or a stale) value."""
    from helion_mlir_backend._compiler.mlir.support.errors import (
        UnsupportedOperationError,
    )

    @helion.kernel(static_shapes=True, backend="mlir")
    def two_phase_bad_return(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out + 0

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    with pytest.raises(UnsupportedOperationError, match="host-prefix return"):
        two_phase_bad_return(x, y)


def test_unresolvable_phase_output_name_raises_clear_diagnostic(monkeypatch):
    """A phase's output tensor must resolve to a plain host variable name to
    be threaded to a later phase/the final return; force the unresolvable
    case (a computed/expression origin) via monkeypatch, since it's not
    reachable through any known real Helion syntax."""
    from helion_mlir_backend._compiler.mlir import phase_plan
    from helion_mlir_backend._compiler.mlir.support.errors import (
        UnsupportedOperationError,
    )

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

    monkeypatch.setattr(phase_plan, "resolve_host_variable_name", lambda hf, t: None)

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    with pytest.raises(UnsupportedOperationError, match="unresolvable phase output"):
        two_phase_kernel(x, y)
