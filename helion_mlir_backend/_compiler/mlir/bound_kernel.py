"""MLIR-aware BoundKernel.compile_config override.

Replaces the Triton codegen + PyCodeCache path with MLIR generation followed
by lighthouse lowering and JIT compilation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Callable

import torch

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

log = logging.getLogger(__name__)


def mlir_compile_config(
    bound_kernel: BoundKernel,
    config: Config | None = None,
    *,
    allow_print: bool = True,
) -> Callable[..., object]:
    """compile_config replacement that generates MLIR and lowers via lighthouse."""
    from helion_mlir_backend._compiler.execution import HelionMLIRExecutor

    if config is None:
        config = bound_kernel._require_implicit_config()
    config = bound_kernel._normalize_config(config)

    if (rv := bound_kernel._compile_cache.get(config)) is not None:
        return rv

    backend = bound_kernel.env.backend
    kernel_name = bound_kernel.kernel.name
    device = bound_kernel.env.device
    hf = bound_kernel.host_function

    from .phase_plan import requires_multi_phase_driver

    tensor_params = [
        (name, value)
        for name, value in hf.params.arguments.items()
        if isinstance(value, torch.Tensor)
    ]
    with bound_kernel.env:
        needs_driver, _ = requires_multi_phase_driver(hf, tensor_params)

    if needs_driver:
        run = _build_multi_phase_driver(bound_kernel, config, device)
    else:
        with bound_kernel.env:
            mlir_module = backend.generate_mlir(hf, config, bound_kernel.env)

        executor = HelionMLIRExecutor(kernel_name=kernel_name, device=device)
        jit_fn = executor.compile(mlir_module)

        # Wrap to match helion's calling convention (args forwarded as-is).
        def run(*args: object) -> object:
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
            return jit_fn(*tensor_args)

    bound_kernel._compile_cache[config] = run
    return run


def _build_multi_phase_driver(
    bound_kernel: BoundKernel,
    config: Config,
    device: torch.device,
) -> Callable[..., object]:
    """Compile one MLIR function per phase and return a callable that drives
    them in order, threading real tensors between phases by host variable
    name (see ``host_prefix.py``/``phase_plan.py``/``codegen.py::
    build_phase_modules`` for the pieces this assembles).
    """
    from .codegen import MLIRModuleBuilder
    from .host_prefix import build_host_prefix_function
    from .support import UnsupportedOperationError
    from helion_mlir_backend._compiler.execution import HelionMLIRExecutor

    hf = bound_kernel.host_function
    with bound_kernel.env:
        builder = MLIRModuleBuilder(hf, config, bound_kernel.env)
        phase_modules = builder.build_phase_modules()

    tensor_param_names = [
        name
        for name, value in hf.params.arguments.items()
        if isinstance(value, torch.Tensor)
    ]
    host_prefix_fn = build_host_prefix_function(hf)
    phase_callables = [
        HelionMLIRExecutor(kernel_name=phase.name, device=device).compile(phase.module)
        for phase in phase_modules
    ]

    return_names = _extract_return_names(hf)

    def run(*args: object) -> object:
        real_args = [a for a in args if isinstance(a, torch.Tensor)]
        names = dict(zip(tensor_param_names, real_args, strict=True))
        names.update(host_prefix_fn(*real_args))

        for phase, jit_fn in zip(phase_modules, phase_callables, strict=True):
            phase_inputs = [names[name] for name in phase.input_names]
            phase_result = jit_fn(*phase_inputs)
            results = phase_result if isinstance(phase_result, list) else [phase_result]
            names.update(zip(phase.output_names, results, strict=True))

        if return_names is None:
            raise UnsupportedOperationError(
                "unsupported host-prefix return statement",
                reason=(
                    "the kernel's own `return` must be a plain name or tuple "
                    "of names for the multi-phase/interop driver to know "
                    "what to return"
                ),
            )
        values = [names[name] for name in return_names]
        return values[0] if len(values) == 1 else values

    return run


def _extract_return_names(hf: HostFunction) -> list[str] | None:
    """The host variable name(s) in the kernel's own trailing ``return``.

    Supports ``return name`` and ``return name1, name2, ...`` only; anything
    else (a computed expression) isn't resolvable this way and returns
    ``None``.
    """
    import ast

    if not hf.body or not isinstance(hf.body[-1], ast.Return):
        return None
    value = hf.body[-1].value
    if value is None:
        return []
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, (ast.Tuple, ast.List)):
        names = [elt.id for elt in value.elts if isinstance(elt, ast.Name)]
        return names if len(names) == len(value.elts) else None
    return None
