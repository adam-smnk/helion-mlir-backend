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

    with bound_kernel.env:
        mlir_module = backend.generate_mlir(
            bound_kernel.host_function, config, bound_kernel.env
        )

    executor = HelionMLIRExecutor(kernel_name=kernel_name, device=device)
    jit_fn = executor.compile(mlir_module)

    # Wrap to match helion's calling convention (args forwarded as-is).
    def run(*args: object) -> object:
        tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
        return jit_fn(*tensor_args)

    bound_kernel._compile_cache[config] = run
    return run
