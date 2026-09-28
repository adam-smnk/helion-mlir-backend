"""Call path of a compiled kernel: host code, argument binding, entry call, return.

Every call runs the kernel's host code up to its device loops (``host_code.py``),
evaluates each entry argument's host expression in the host locals, calls the
memref-ABI entry, then finishes the host code and returns its return value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..execution import compile_entry
from .codegen import MLIRModuleBuilder
from .host_code import build_host_function
from .support import UnsupportedOperationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

    from ..execution import CompiledEntry
    from ..execution import EntryArg


def mlir_compile_config(
    bound_kernel: BoundKernel,
    config: Config | None = None,
    *,
    allow_print: bool = True,
) -> Callable[..., object]:
    """``BoundKernel.compile_config`` for the MLIR backend (one module per config)."""
    if config is None:
        config = bound_kernel._require_implicit_config()
    config = bound_kernel._normalize_config(config)
    if (run := bound_kernel._compile_cache.get(config)) is not None:
        return run
    run = compile_kernel(bound_kernel.host_function, config, bound_kernel.env)
    bound_kernel._compile_cache[config] = run
    return run


def compile_kernel(
    hf: HostFunction,
    config: Config,
    env: CompileEnvironment,
    *,
    pipeline: str | None = None,
) -> Callable[..., object]:
    """A callable with the kernel's own signature and Helion call semantics."""
    with env:
        builder = MLIRModuleBuilder(hf, config, env)
        module = builder.build()
        host_function = build_host_function(
            hf, builder.context.geometry.block_size, config
        )
    entry = compile_entry(module, hf.name, pipeline=pipeline)
    host_globals = host_function.__globals__
    arg_exprs = [
        compile(arg.name, f"<helion-mlir-arg:{arg.name}>", "eval") for arg in entry.args
    ]

    def run(*args: object) -> object:
        host = host_function(*args)
        try:
            local_vars = next(host)
            call_entry(
                entry, [eval(expr, host_globals, local_vars) for expr in arg_exprs]
            )
            next(host)
        except StopIteration as done:
            return done.value
        raise AssertionError("the host function yields once")

    return run


def call_entry(entry: CompiledEntry, values: list[object]) -> list[torch.Tensor]:
    """Call ``entry`` with one value per argument; returns the inout tensors.

    Tensors are passed contiguous (a copy for strided ones, copied back for inouts).
    An input that overlaps an inout is cloned so the ``restrict`` arguments hold;
    overlapping inouts are rejected.
    """
    tensors = [
        _checked_tensor(arg, value)
        for arg, value in zip(entry.args, values, strict=True)
        if arg.role != "scalar"
    ]
    roles = [arg.role for arg in entry.args if arg.role != "scalar"]
    inouts = [t for t, role in zip(tensors, roles, strict=True) if role == "inout"]
    for i, first in enumerate(inouts):
        for second in inouts[i + 1 :]:
            if _overlaps(first, second):
                raise UnsupportedOperationError(
                    "kernel writes two tensors that share memory",
                    reason="each written tensor must be its own buffer",
                )
    staged = [
        tensor.clone(memory_format=torch.contiguous_format)
        if role == "in" and any(_overlaps(tensor, other) for other in inouts)
        else tensor.contiguous()
        for tensor, role in zip(tensors, roles, strict=True)
    ]
    scalars = [
        torch.tensor(value, dtype=arg.dtype)
        for arg, value in zip(entry.args, values, strict=True)
        if arg.role == "scalar"
    ]
    entry([*staged, *scalars])
    for tensor, copy, role in zip(tensors, staged, roles, strict=True):
        if role == "inout" and copy is not tensor:
            tensor.copy_(copy)
    return inouts


def _checked_tensor(arg: EntryArg, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"'{arg.name}' must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.device.type != "cpu":
        raise NotImplementedError(
            f"'{arg.name}' is on {value.device}; only CPU is supported"
        )
    if tuple(value.shape) != arg.shape or value.dtype != arg.dtype:
        raise ValueError(
            f"'{arg.name}' has shape {tuple(value.shape)} and dtype {value.dtype}; the "
            f"compiled kernel expects {arg.shape} and {arg.dtype}"
        )
    return value


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    if a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr():
        return False
    (a_start, a_end), (b_start, b_end) = _byte_span(a), _byte_span(b)
    return a_start < b_end and b_start < a_end


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    last = sum(
        (size - 1) * stride for size, stride in zip(t.shape, t.stride(), strict=True)
    )
    return (
        t.storage_offset() * t.element_size(),
        (t.storage_offset() + last + 1) * t.element_size(),
    )
