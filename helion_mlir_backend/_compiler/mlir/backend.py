"""MLIR backend class for Helion.

Registers as a Helion compiler backend named "mlir".  The backend:

- Reuses Helion's backend-neutral compilation pipeline (type propagation,
    device IR construction, etc.).
- Replaces the final code-generation step with an MLIR module builder that
  produces Linalg-on-Tensors IR instead of Triton Python source code.

The generated MLIR is intentionally high-level: Linalg-on-Tensors phase
functions plus a thin memref entry function that fixes the calling convention
(no tiling or lowering to LLVM), so a downstream MLIR compiler can apply its
own tiling, vectorization, and memory-placement passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion import exc
from helion._compiler.backend import Backend

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction


class MLIRBackend(Backend):
    """Helion backend that emits MLIR Linalg-on-Tensors IR.

    Compilation (parsing, type propagation, device IR construction) uses
    Helion's backend-neutral pipeline. Only the final codegen step is replaced.
    """

    @property
    def name(self) -> str:
        return "mlir"

    @property
    def experimental(self) -> bool:
        return True

    def autotune(
        self,
        bound_kernel: object,
        args: object,
        *,
        force: bool = False,
        **kwargs: object,
    ) -> object:
        # CPU has no hardware cache key; skip autotuning and use default config.
        return bound_kernel.env.config_spec.default_config()

    def dtype_str(self, dtype: object) -> str:
        import torch

        dtype_names = {
            torch.bool: "torch.bool",
            torch.float16: "torch.float16",
            torch.bfloat16: "torch.bfloat16",
            torch.float32: "torch.float32",
            torch.float64: "torch.float64",
            torch.int8: "torch.int8",
            torch.int16: "torch.int16",
            torch.int32: "torch.int32",
            torch.int64: "torch.int64",
            torch.uint8: "torch.uint8",
        }
        return dtype_names.get(dtype, str(dtype))

    def acc_type(self, dtype: object) -> str:
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        raise exc.BackendUnsupported(self.name, "Python function decorators")

    @property
    def constexpr_type(self) -> str:
        raise exc.BackendUnsupported(self.name, "Python constexpr annotations")

    @property
    def default_launcher_name(self) -> str:
        raise exc.BackendUnsupported(self.name, "Python launchers")

    @property
    def library_imports(self) -> dict[str, str]:
        raise exc.BackendUnsupported(self.name, "Python library imports")

    # ------------------------------------------------------------------
    # MLIR generation entry point
    # ------------------------------------------------------------------

    def generate_mlir(
        self,
        host_function: HostFunction,
        config: object,
        env: CompileEnvironment,
    ) -> object:
        """Build and return an ``mlir.ir.Module`` for the compiled kernel.

        Parameters
        ----------
        host_function:
            The :class:`~helion._compiler.host_function.HostFunction` produced
            by :func:`~helion._compiler.kernel_compiler.KernelCompiler.compile`.
        config:
            The :class:`~helion.runtime.config.Config` specifying block sizes
            and other tuning parameters.
        env:
            The :class:`~helion._compiler.compile_environment.CompileEnvironment`
            that was active during compilation.

        Returns
        -------
        mlir.ir.Module
            Parsed and verified MLIR module containing a ``func.func`` with
            Linalg-on-Tensors IR equivalent to the Helion kernel.
        """
        from .codegen import MLIRModuleBuilder

        builder = MLIRModuleBuilder(host_function, config, env)
        return builder.build()

    # ------------------------------------------------------------------
    # MLIR execution entry point (experimental)
    # ------------------------------------------------------------------

    def execute_mlir(
        self,
        mlir_module: object,
        *input_tensors: object,
        kernel_name: str = "kernel",
        pipeline: str | None = None,
    ) -> object:
        """Execute a module from :meth:`generate_mlir` via lighthouse (consumes it).

        ``input_tensors`` are the kernel's tensor parameters in declaration order.
        No host code runs: tensors the kernel creates on the host and writes are
        zero-initialized, and kernels that read host-computed tensors or take
        runtime scalars are rejected. Use ``compile_mlir`` or a direct kernel call
        for full Helion semantics.

        Returns every tensor the kernel writes, in entry argument order: one
        tensor, or a list when there are several.
        """
        import torch

        from ..execution import compile_entry
        from .driver import call_entry
        from .support import UnsupportedOperationError

        if not input_tensors or not all(
            isinstance(tensor, torch.Tensor) for tensor in input_tensors
        ):
            raise TypeError("input_tensors must be non-empty torch.Tensor instances")

        device = input_tensors[0].device
        if any(tensor.device != device for tensor in input_tensors[1:]):
            raise ValueError("all input tensors must be on the same device")
        if device.type != "cpu":
            raise NotImplementedError(
                f"Only CPU device supported for execution; got {device.type}"
            )

        entry = compile_entry(mlir_module, kernel_name, pipeline=pipeline)
        values: list[object] = []
        for arg in entry.args:
            if arg.tensor_param is not None:
                values.append(input_tensors[arg.tensor_param])
            elif arg.role == "inout":
                values.append(torch.zeros(arg.shape, dtype=arg.dtype))
            else:
                raise UnsupportedOperationError(
                    f"execute_mlir cannot supply host value '{arg.name}'",
                    reason="execute_mlir runs no host code",
                    alternatives=[
                        "helion_mlir_backend.compile_mlir(kernel, args)",
                        "call the kernel with @helion.kernel(backend='mlir')",
                    ],
                )
        outputs = call_entry(entry, values)
        return outputs[0] if len(outputs) == 1 else outputs
