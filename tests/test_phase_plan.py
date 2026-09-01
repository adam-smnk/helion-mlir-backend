"""Unit tests for the phase/host-tensor analysis utilities (phase_plan.py).

Read-only analysis, not yet wired into codegen/execution -- see host_prefix.py
and session memory for the follow-up wiring steps.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from helion_mlir_backend._compiler.mlir.phase_plan import build_phase_plans
from helion_mlir_backend._compiler.mlir.phase_plan import find_extra_host_tensor_names


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


def _declared_param_names(hf) -> set[str]:
    return {
        name
        for name, value in hf.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }


def test_single_phase_kernel_has_one_plan_with_no_extra_inputs():
    @helion.kernel(static_shapes=True)
    def single_phase(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = x[tm, tn] + y[tm, tn]
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    hf = _compile_host_function(single_phase.fn, [x, y])
    declared = _declared_param_names(hf)

    assert len(hf.device_ir.phases) == 1
    extras = find_extra_host_tensor_names(hf, declared)
    # `out` is the store destination only (never read), not a real dependency.
    assert extras == ["out"]

    (plan,) = build_phase_plans(hf, declared, extras)
    assert plan.input_names == ["x", "y"]
    assert len(plan.outputs) == 1


def test_two_phase_barrier_kernel_threads_cross_phase_tensor_by_name():
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
    hf = _compile_host_function(two_phase_kernel.fn, [x, y])
    declared = _declared_param_names(hf)

    assert len(hf.device_ir.phases) == 2
    extras = find_extra_host_tensor_names(hf, declared)
    assert set(extras) == {"mid", "out"}

    plan0, plan1 = build_phase_plans(hf, declared, extras)
    assert plan0.input_names == ["x", "y"]
    assert plan1.input_names == ["mid"]
    assert "out" not in plan0.input_names
    assert "out" not in plan1.input_names


def test_extra_host_tensor_from_real_op_is_discovered():
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
    declared = _declared_param_names(hf)

    extras = find_extra_host_tensor_names(hf, declared)
    assert "scale" in extras

    (plan,) = build_phase_plans(hf, declared, extras)
    assert "scale" in plan.input_names


def test_two_phase_kernel_with_nested_reduction_before_barrier():
    """Regression test: a phase's root graph id can differ from its position
    in `KernelPhase.roots` (a nested-reduction loop body is registered as its
    own graph before a later root), so `_iter_phase_graphs` must resolve
    through `device_ir.root_ids[position]`, not treat the position as a
    graph id directly.
    """

    @helion.kernel(static_shapes=True)
    def two_phase_with_reduction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        k2, n = y.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            mid[tm, tn] = acc
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out

    x = torch.randn(8, 16)
    y = torch.randn(16, 8)
    hf = _compile_host_function(two_phase_with_reduction.fn, [x, y])
    declared = _declared_param_names(hf)

    # A nested for-loop body graph is registered before the second root, so
    # root graph ids (e.g. [1, 2]) diverge from root positions (e.g. [0, 1]).
    assert hf.device_ir.root_ids != list(range(len(hf.device_ir.root_ids)))

    extras = find_extra_host_tensor_names(hf, declared)
    plan0, plan1 = build_phase_plans(hf, declared, extras)
    assert plan0.input_names == ["x", "y"]
    assert plan1.input_names == ["mid"]
