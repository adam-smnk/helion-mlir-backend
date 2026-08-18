"""Execution pipeline for MLIR kernels via lighthouse.

Provides:
1. MLIR inlining to eliminate function calls
2. Lighthouse-based lowering to LLVM IR
3. JIT compilation and tensor marshalling
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING
from typing import Callable

import torch

if TYPE_CHECKING:
    import mlir.ir as ir

log = logging.getLogger(__name__)


def _dump_if(envvar: str, label: str, module: ir.Module) -> None:
    """Print *module* to stdout when *envvar* is set to a truthy value."""
    if os.environ.get(envvar, "").strip() not in ("", "0", "false", "no"):
        print(f"=== {label} ===", flush=True)
        print(module, flush=True)


# ---------------------------------------------------------------------------
# Phase 1: MLIR Inlining
# ---------------------------------------------------------------------------


def inline_module(module: ir.Module) -> ir.Module:
    """Inline all function calls in an MLIR module.

    Attempts to inline outlined functions. If MLIR passes are not available,
    returns the module unchanged and relies on lighthouse to handle function calls.

    Parameters
    ----------
    module : ir.Module
        MLIR module with outlined functions (e.g. @_aten_0, @_aten_1, etc.)

    Returns
    -------
    ir.Module
        Module with inlined calls (or original if passes unavailable).

    Raises
    ------
    RuntimeError
        If inlining fails critically.
    """

    import mlir.ir as ir
    from mlir.passmanager import PassManager

    with module.context, ir.Location.unknown():
        pm = PassManager.parse("builtin.module(inline,canonicalize)")
        pm.run(module.operation)

    log.info("MLIR inlining completed")
    return module


# ---------------------------------------------------------------------------
# Phase 2 & 3: Lighthouse Lowering + Tensor Marshalling
# ---------------------------------------------------------------------------


@dataclass
class BufferMetadata:
    """Output buffer metadata for result allocation."""

    shape: list[int]
    dtype: torch.dtype
    device: torch.device


class HelionMLIRExecutor:
    """Execute a Helion-generated MLIR kernel via lighthouse.

    Handles inlining, lighthouse lowering, JIT compilation, and
    tensor marshalling following lighthouse's external-buffer model.

    Parameters
    ----------
    kernel_name : str
        Name of the public kernel function (e.g. "matmul_kernel").
    device : torch.device
        Target device (currently CPU only).
    """

    def __init__(
        self,
        kernel_name: str = "kernel",
        device: torch.device | None = None,
    ) -> None:
        self.kernel_name = kernel_name
        self.device = device or torch.device("cpu")
        if self.device.type != "cpu":
            raise NotImplementedError(
                f"Only CPU device supported; got {self.device.type}"
            )
        self._jit_fn: Callable | None = None
        # Populated during _inline_phase; consumed by _compile_phase
        self._result_metadata: list[BufferMetadata] = []

    def compile(
        self, mlir_module: ir.Module
    ) -> Callable[..., torch.Tensor | list[torch.Tensor]]:
        """Inline, lower, and JIT-compile without executing; returns a reusable callable."""
        try:
            inlined = self._inline_phase(mlir_module)
        except Exception as exc:
            raise RuntimeError(f"MLIR inlining failed: {exc}") from exc

        _dump_if(
            "HELION_MLIR_DUMP_PRE_LOWERING", "MLIR before lighthouse lowering", inlined
        )

        try:
            lowered = self._lighthouse_lower_phase(inlined)
        except Exception as exc:
            raise RuntimeError(
                f"Lighthouse lowering failed: {exc}. "
                "This may indicate unsupported MLIR constructs or limitation in lighthouse. "
                "Check MLIR module structure and kernel complexity."
            ) from exc

        try:
            return self._compile_phase(lowered)
        except Exception as exc:
            raise RuntimeError(f"JIT compilation failed: {exc}") from exc

    def prepare_and_execute(
        self,
        mlir_module: ir.Module,
        *input_tensors: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Inline, lower, JIT-compile, and execute the kernel.

        Parameters
        ----------
        mlir_module : ir.Module
            Generated MLIR module with outlined ATen helpers.
        *input_tensors : torch.Tensor
            Input tensors to the kernel (must match kernel signature).

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Computed result(s).

        Raises
        ------
        RuntimeError
            If inlining, lowering, or execution fails.
        """
        jit_fn = self.compile(mlir_module)

        try:
            return jit_fn(*input_tensors)
        except Exception as exc:
            raise RuntimeError(f"Kernel execution failed: {exc}") from exc

    def _inline_phase(self, module: ir.Module) -> ir.Module:
        """Phase 1: Inline and extract output metadata while types are still tensors."""
        _dump_if("HELION_MLIR_DUMP_IR", "MLIR before inlining", module)
        inlined = inline_module(module)
        self._result_metadata = self._extract_result_metadata_pre_lowering(
            inlined, self.kernel_name
        )
        return inlined

    def _extract_result_metadata_pre_lowering(
        self,
        module: ir.Module,
        kernel_name: str,
    ) -> list[BufferMetadata]:
        """Extract output shapes/dtypes from the func.func IR before lowering."""
        import mlir.ir as ir

        from helion_mlir_backend._compiler.mlir.support.type_utils import (
            mlir_dtype_to_torch,
        )

        func_op = None
        for op in module.operation.regions[0].blocks[0].operations:
            name_attr = op.attributes.get("sym_name")
            if name_attr is not None:
                op_name = (
                    name_attr.value
                    if hasattr(name_attr, "value")
                    else str(name_attr).strip('"')
                )
                if op_name == kernel_name:
                    func_op = op
                    break

        if func_op is None:
            log.warning("Entry function not found for metadata extraction")
            return []

        results: list[BufferMetadata] = []
        for res_type in func_op.type.results:
            if not isinstance(res_type, ir.RankedTensorType):
                continue
            elem = str(res_type.element_type)
            dtype = mlir_dtype_to_torch(elem)
            results.append(
                BufferMetadata(
                    shape=list(res_type.shape),
                    dtype=dtype,
                    device=self.device,
                )
            )

        log.info(f"Extracted {len(results)} output(s) from pre-lowering IR")
        return results

    def _lighthouse_lower_phase(self, module: ir.Module) -> ir.Module:
        """Phase 2: Lower to LLVM IR using lighthouse pipeline."""
        import mlir.ir as ir

        try:
            from lighthouse.pipeline.descriptor import Descriptor
            from lighthouse.pipeline.driver import BackendDriver
        except ImportError as exc:
            raise ImportError(
                "lighthouse is required for kernel lowering. "
                "Install via: pip install lighthouse"
            ) from exc

        with module.context, ir.Location.unknown():
            driver = BackendDriver(
                module,
                self.kernel_name,
                result_to_args=True,
                benchmark=False,
            )
            if os.environ.get("HELION_MLIR_PIPELINE", "").strip() == "1":
                driver.add_stage(
                    Descriptor("./pipeline.yaml", base_path=os.path.dirname(__file__))
                )
            else:
                driver.add_stage(Descriptor("scalar-lowering.yaml"))
            lowered = driver.apply(module)

        log.info("Lowered via scalar-lowering pipeline")
        _dump_if("HELION_MLIR_DUMP_LOWERED", "MLIR after lighthouse lowering", lowered)
        return lowered

    def _compile_phase(
        self,
        lowered_module: ir.Module,
    ) -> Callable:
        """Phase 3: Compile lowered LLVM IR to a JIT callable.

        Uses the _mlir_ciface_ C-interface wrapper (void return, output as first
        arg) so the Runner receives a well-defined calling convention.  Output
        tensors are pre-allocated from metadata extracted before lowering.
        """
        try:
            from lighthouse.execution.runner import Runner
            from lighthouse.ingress.torch.compile import TorchMemoryManager
        except ImportError as exc:
            raise ImportError(
                "lighthouse execution backend is required. "
                "Install via: pip install lighthouse"
            ) from exc

        # result_to_args=True means matmul_kernel now has void return and takes
        # outputs as trailing memref arguments — use it directly.
        entry_func = self.kernel_name
        log.info(f"JIT entry: {entry_func}")

        result_meta = self._result_metadata
        runner = Runner(
            lowered_module,
            mem_manager_cls=TorchMemoryManager,
            shared_libs=[],
        )

        def jit_wrapper(
            *input_tensors: torch.Tensor,
        ) -> torch.Tensor | list[torch.Tensor]:
            # Pre-allocate contiguous output buffers.
            outputs = [
                torch.empty(
                    m.shape,
                    dtype=m.dtype,
                    device=m.device,
                    memory_format=torch.contiguous_format,
                )
                for m in result_meta
            ]
            # TorchMemoryManager ciface convention: output first, then inputs.
            runner.execute(entry_func, [*outputs, *input_tensors])
            return outputs[0] if len(outputs) == 1 else outputs

        self._jit_fn = jit_wrapper
        return jit_wrapper
