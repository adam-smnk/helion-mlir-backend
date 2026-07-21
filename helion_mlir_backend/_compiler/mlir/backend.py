"""MLIR backend class for Helion.

Registers as a Helion compiler backend named "mlir".  The backend:

- Inherits most of the compilation pipeline from TritonBackend (type
  propagation, device IR construction, etc.) since these are largely
  backend-agnostic.
- Replaces the final code-generation step with an MLIR module builder that
  produces Linalg-on-Tensors IR instead of Triton Python source code.

The generated MLIR is intentionally high-level (no bufferization, no lowering
to LLVM) so that a downstream MLIR compiler can apply its own tiling,
vectorization, and memory-placement passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion._compiler.triton.backend import TritonBackend

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction
    from helion._compiler.compile_environment import CompileEnvironment


class MLIRBackend(TritonBackend):
    """Helion backend that emits MLIR Linalg-on-Tensors IR.

    Compilation (parsing, type propagation, device IR construction) reuses the
    TritonBackend pipeline.  Only the final codegen step is replaced.
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
