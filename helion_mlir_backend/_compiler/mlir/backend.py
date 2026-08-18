"""MLIR backend class for Helion.

Registers as a Helion compiler backend named "mlir".  The backend:

- Reuses Helion's backend-neutral compilation pipeline (type propagation,
    device IR construction, etc.).
- Replaces the final code-generation step with an MLIR module builder that
  produces Linalg-on-Tensors IR instead of Triton Python source code.

The generated MLIR is intentionally high-level (no bufferization, no lowering
to LLVM) so that a downstream MLIR compiler can apply its own tiling,
vectorization, and memory-placement passes.
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

    @property
    def codegen_name(self) -> str:
        # Fall back to triton codegen for any per-op registrations we haven't
        # overridden yet.  This keeps the compilation pipeline working even
        # before all MLIR op lowerings are implemented.
        return "triton"

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
    ) -> object:
        """Execute a Helion-generated MLIR kernel via lighthouse.

        Preprocesses the module to move results to arguments (following
        lighthouse calling convention), then inlines, lowers, JIT-compiles,
        and executes the kernel.

        Parameters
        ----------
        mlir_module : ir.Module
            Generated MLIR module from :meth:`generate_mlir`.
        *input_tensors : torch.Tensor
            Input tensors matching the kernel signature.
        kernel_name : str
            Name of the public kernel function (default: "kernel").

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Computed result(s).

        Raises
        ------
        RuntimeError
            If preprocessing, inlining, lowering, compilation, or execution fails.
        NotImplementedError
            If device is not CPU.
        """
        import torch

        if not isinstance(input_tensors[0], torch.Tensor):
            raise TypeError("input_tensors must be torch.Tensor instances")

        device = input_tensors[0].device
        if device.type != "cpu":
            raise NotImplementedError(
                f"Only CPU device supported for execution; got {device.type}"
            )

        # Execute via lighthouse; result_to_args is handled inside the executor.
        from helion_mlir_backend._compiler.execution import HelionMLIRExecutor

        executor = HelionMLIRExecutor(kernel_name=kernel_name, device=device)
        return executor.prepare_and_execute(mlir_module, *input_tensors)
