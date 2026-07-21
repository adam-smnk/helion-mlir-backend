"""Public MLIR API exposed via ``helion.mlir`` shim.

This mirrors the historical ``helion.mlir.generate_mlir`` entrypoint while
keeping implementation out-of-tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir


def generate_mlir(
    kernel: object,
    args: list[object],
    *,
    config: object | None = None,
) -> ir.Module:
    """Lower a Helion kernel to an MLIR Linalg-on-Tensors module."""
    try:
        import mlir.ir  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mlir-python-bindings is required for MLIR code generation. "
            "Install it with: pip install mlir-python-bindings"
        ) from exc

    from helion._compiler.backend_registry import get_backend_class
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin
    from helion.runtime.settings import Settings

    fn = getattr(kernel, "_fn", None) or getattr(kernel, "fn", None)
    if fn is None:
        raise ValueError(
            "Expected a @helion.kernel-decorated function; "
            f"got {type(kernel).__name__}"
        )

    settings: Settings = getattr(kernel, "_settings", None) or getattr(
        kernel, "settings", Settings()
    )
    settings.backend = "mlir"

    if config is None:
        config = getattr(kernel, "_default_config", None)

    import torch

    device = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            device = arg.device
            break
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = CompileEnvironment(device, settings)

    with env:
        fake_args: list[object] = []
        sig_params = list(kernel.signature.parameters)
        for name, arg in zip(sig_params, args, strict=False):
            fake_args.append(env.to_fake(arg, ArgumentOrigin(name)))

        compiler = KernelCompiler(env)
        host_function = compiler.compile(fn, fake_args, {})

        if config is None:
            config = env.config_spec.default_config()

        backend = get_backend_class("mlir")()
        module = backend.generate_mlir(host_function, config, env)

    return module
