"""Lower a generated module with lighthouse and JIT its memref-ABI entry function.

:func:`compile_entry` returns a :class:`CompiledEntry` that takes the entry's
buffers in argument order. Failures keep their exception type and get a note
naming the stage. Compiled entries are cached in-process by module text and
pipeline, so equal modules (e.g. configs that clamp to the same tiles) JIT once.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from typing import TYPE_CHECKING

from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch.compile import TorchMemoryManager
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver
import mlir.ir as ir
from mlir.passmanager import PassManager

from helion_mlir_backend._compiler.mlir.in_place import update_carried_values_in_place
from helion_mlir_backend._compiler.mlir.support.debug import DebugOptions
from helion_mlir_backend._compiler.mlir.support.debug import use_optimizing_pipeline
from helion_mlir_backend._compiler.mlir.support.type_utils import mlir_dtype_to_torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch

PIPELINES = ("scalar", "opt")
_JIT_CACHE: OrderedDict[tuple[str, str], CompiledEntry] = OrderedDict()
_JIT_CACHE_SIZE = 128


def _dump_if(enabled: bool, label: str, module: ir.Module) -> None:
    if enabled:
        print(f"=== {label} ===", flush=True)
        print(module, flush=True)


def pipeline_descriptor(pipeline: str | None = None) -> Descriptor:
    """The lighthouse pipeline by name; by default ``opt`` if
    ``HELION_MLIR_PIPELINE=1``, else ``scalar``."""
    if pipeline is None:
        pipeline = "opt" if use_optimizing_pipeline() else "scalar"
    if pipeline == "opt":
        return Descriptor("./pipeline.yaml", base_path=os.path.dirname(__file__))
    if pipeline == "scalar":
        return Descriptor("./scalar.yaml", base_path=os.path.dirname(__file__))
    raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {PIPELINES}")


def inline_module(module: ir.Module) -> ir.Module:
    """Inline phase and helper functions into the entry and update loop-carried
    values in place."""
    with module.context, ir.Location.unknown():
        PassManager.parse("builtin.module(inline,canonicalize)").run(module.operation)
        update_carried_values_in_place(module)
        PassManager.parse("builtin.module(canonicalize)").run(module.operation)
    return module


@dataclass(frozen=True)
class EntryArg:
    """One argument of the entry function, from its ``helion.*`` argument attributes."""

    name: str
    """The host expression that produces the argument."""
    role: str
    """``in``, ``inout`` or ``scalar``."""
    tensor_param: int | None
    shape: tuple[int, ...]
    """``-1`` for a size only known at run time."""
    dtype: torch.dtype


@dataclass(frozen=True)
class CompiledEntry:
    name: str
    args: tuple[EntryArg, ...]
    runner: Runner

    def __call__(self, buffers: list[torch.Tensor]) -> None:
        with _stage(f"running '{self.name}'"):
            self.runner.execute(self.name, buffers)


def entry_args(module: ir.Module, entry: str) -> tuple[EntryArg, ...]:
    """The entry function's arguments, read before lowering."""
    fn = next(
        (
            op
            for op in module.body.operations
            if "sym_name" in op.attributes
            and ir.StringAttr(op.attributes["sym_name"]).value == entry
        ),
        None,
    )
    if fn is None:
        raise ValueError(f"no function named '{entry}' in the module")
    arg_types = ir.FunctionType(
        ir.TypeAttr(fn.attributes["function_type"]).value
    ).inputs
    arg_attrs = fn.attributes.get("arg_attrs")
    if arg_attrs is None:
        raise ValueError(f"'{entry}' is not a Helion MLIR entry function")
    args = []
    for arg_type, attrs in zip(arg_types, ir.ArrayAttr(arg_attrs), strict=True):
        attrs = ir.DictAttr(attrs)
        memref = ir.MemRefType(arg_type)
        dtype = mlir_dtype_to_torch(str(memref.element_type))
        if dtype is None:
            raise ValueError(
                f"'{entry}' takes a memref of unsupported {memref.element_type}"
            )
        args.append(
            EntryArg(
                ir.StringAttr(attrs["helion.name"]).value,
                ir.StringAttr(attrs["helion.role"]).value,
                ir.IntegerAttr(attrs["helion.param"]).value
                if "helion.param" in attrs
                else None,
                tuple(
                    -1 if ir.ShapedType.is_dynamic_size(size) else size
                    for size in memref.shape
                ),
                dtype,
            )
        )
    return tuple(args)


def compile_entry(
    module: ir.Module, entry: str, *, pipeline: str | None = None
) -> CompiledEntry:
    """Inline, lower and JIT-compile ``entry`` (consumes ``module``), or reuse the
    entry compiled from an identical module with the same pipeline."""
    if pipeline is None:
        pipeline = "opt" if use_optimizing_pipeline() else "scalar"
    debug = DebugOptions.from_env()
    if debug.dump_ir or debug.dump_pre_lowering or debug.dump_lowered:
        return _compile_entry(module, entry, pipeline, debug)
    key = (hashlib.sha256(str(module).encode()).hexdigest(), pipeline)
    if (compiled := _JIT_CACHE.get(key)) is not None:
        _JIT_CACHE.move_to_end(key)
        return compiled
    compiled = _compile_entry(module, entry, pipeline, debug)
    _JIT_CACHE[key] = compiled
    if len(_JIT_CACHE) > _JIT_CACHE_SIZE:
        _JIT_CACHE.popitem(last=False)
    return compiled


def _compile_entry(
    module: ir.Module, entry: str, pipeline: str, debug: DebugOptions
) -> CompiledEntry:
    args = entry_args(module, entry)
    _dump_if(debug.dump_ir, "MLIR before inlining", module)
    with _stage("inlining"):
        inline_module(module)
    _dump_if(debug.dump_pre_lowering, "MLIR before lighthouse lowering", module)
    descriptor = pipeline_descriptor(pipeline)
    with (
        _stage(f"lowering with the lighthouse '{pipeline}' pipeline"),
        module.context,
        ir.Location.unknown(),
    ):
        driver = BackendDriver(module, entry, result_to_args=False, benchmark=False)
        driver.add_stage(descriptor)
        lowered = driver.apply(module)
    _dump_if(debug.dump_lowered, "MLIR after lighthouse lowering", lowered)
    with _stage("JIT compilation"):
        runner = Runner(lowered, mem_manager_cls=TorchMemoryManager, shared_libs=[])
    return CompiledEntry(entry, args, runner)


@contextmanager
def _stage(what: str) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        exc.add_note(f"Helion MLIR backend: failed while {what}")
        raise
