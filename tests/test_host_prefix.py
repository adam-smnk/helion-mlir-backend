"""Unit tests for the host-prefix driver primitive (host_prefix.py).

Covers the mechanism in isolation (not yet wired into the kernel-call path --
see host_tensor_interop plan in session memory for the follow-up steps that
consume this).
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from helion_mlir_backend._compiler.mlir.host_prefix import UnsupportedHostPrefixError
from helion_mlir_backend._compiler.mlir.host_prefix import build_host_prefix_function


def _compile_host_function(fn, args):
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin
    from helion.runtime.settings import Settings

    settings = Settings()
    settings.backend = "mlir"
    env = CompileEnvironment(args[0].device, settings)
    with env:
        fake_args = [
            env.to_fake(arg, ArgumentOrigin(f"a{i}")) for i, arg in enumerate(args)
        ]
        compiler = KernelCompiler(env)
        return compiler.compile(fn, fake_args, {})


def test_extra_host_tensor_is_recomputed_for_real():
    @helion.kernel(static_shapes=True)
    def host_side_scale(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        scale = x.mean() * 100.0
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            s = hl.load(scale, [])
            out[tm, tn] = x[tm, tn] + s
        return out

    x = torch.randn(8, 8)
    hf = _compile_host_function(host_side_scale.fn, [x])
    prefix_fn = build_host_prefix_function(hf)
    result = prefix_fn(x)

    assert "scale" in result
    torch.testing.assert_close(result["scale"], x.mean() * 100.0)


def test_host_for_loop_is_preserved_and_reexecuted():
    @helion.kernel(static_shapes=True)
    def with_host_loop(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        total = 0
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for i in range(3):
            total = total + i
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn]
        return out

    x = torch.randn(4, 4)
    hf = _compile_host_function(with_host_loop.fn, [x])
    prefix_fn = build_host_prefix_function(hf)
    result = prefix_fn(x)

    assert result["total"] == 3


def test_grid_loop_body_does_not_execute_in_host_prefix():
    @helion.kernel(static_shapes=True)
    def simple_copy(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn] + 1.0
        return out

    x = torch.randn(4, 4)
    hf = _compile_host_function(simple_copy.fn, [x])
    prefix_fn = build_host_prefix_function(hf)
    result = prefix_fn(x)

    # The device loop is neutralized, so `out` stays at its host-allocated
    # (zero) value -- the actual compute happens in the separately compiled
    # MLIR phase function, not here.
    torch.testing.assert_close(result["out"], torch.zeros(4, 4))
    torch.testing.assert_close(
        result["_helion_mlir_host_prefix_return_value"], torch.zeros(4, 4)
    )


def test_early_return_is_rejected():
    @helion.kernel(static_shapes=True)
    def early_return(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        if m == 0:
            return out
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn]
        return out

    x = torch.randn(4, 4)
    hf = _compile_host_function(early_return.fn, [x])
    with pytest.raises(UnsupportedHostPrefixError):
        build_host_prefix_function(hf)
