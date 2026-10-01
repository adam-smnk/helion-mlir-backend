# MLIR Backend Usage Guide

This guide describes the current usage of the MLIR backend in this repository.

## Current Backend State

- The backend is experimental, but it is not IR-only.
- End-to-end CPU execution is supported through the lighthouse execution path.
- Three user-facing flows are validated in tests:
  - Direct flow: call a kernel decorated with `backend="mlir"`.
  - `compile_mlir(kernel, args)`: the same call semantics as a standalone callable.
  - Explicit flow: `generate_mlir` then `MLIRBackend.execute_mlir`, which runs no
    host code (see `docs/MLIR_LIMITATIONS.md` item 11).

The direct flow and `compile_mlir` run the kernel's host code on every call and
return the kernel's own `return` value, like Helion's Triton wrapper. Reference
tests live under `tests/` (for the call semantics, `tests/test_calling_convention.py`
and `tests/test_multi_phase_execution.py`). Runnable usage examples live under
`examples/` at the repository root.

## Recommended Environment

Use the uv-managed environment for this project:

```bash
uv sync
uv run pytest -q tests/test_mlir_execution.py
```

Running with a non-uv interpreter can miss required dependencies (for example torch-mlir packages).

## Execution Paths

### 1) Explicit MLIR generation and execution

```python
import torch
import helion
import helion.language as hl
from helion_mlir_backend import generate_mlir


def _backend():
    from helion._compiler.backend_registry import get_backend_class

    return get_backend_class("mlir")()


@helion.kernel(static_shapes=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


A = torch.randn(32, 32)
B = torch.randn(32, 32)

mlir_module = generate_mlir(add_kernel, [A, B])
C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
```

### 2) Direct kernel call with backend="mlir"

```python
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=True, backend="mlir")
def add_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


A = torch.randn(32, 32)
B = torch.randn(32, 32)
C = add_direct(A, B)
```

### 3) `compile_mlir`

```python
from helion_mlir_backend import compile_mlir

add = compile_mlir(add_kernel, [A, B], pipeline="scalar")  # default: "opt"
C = add(A, B)
```

The generated module holds one private tensor function per phase and a public
memref-ABI entry named after the kernel; `docs/MLIR_DESIGN.md` (Calling
Convention) describes its arguments.

## Kernel Requirements

### Required

- `static_shapes=True` gives fully static IR; `static_shapes=False` compiles one
  entry per shape bucket that takes runtime sizes (see `docs/MLIR_LIMITATIONS.md`).
- Provide tensor type annotations.
- Place tensor work inside helion device loops (`hl.tile(...)`).

### Nested reduction pattern (matmul-like)

```python
@helion.kernel(static_shapes=True, backend="mlir", config=helion.Config(block_sizes=[16, 8, 32]))
def matmul_tiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    k2, n = y.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
```

Equivalent accumulation style is also supported:

```python
acc = acc + torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
```

### Multi-phase kernels (`hl.barrier()`) and host-tensor interop

Supported by every flow; `execute_mlir` rejects kernels that read
host-computed tensors (it runs no host code):

```python
@helion.kernel(static_shapes=True, backend="mlir")
def two_phase(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    scale = x.mean() * 2.0  # host tensor, not a declared parameter
    mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        s = hl.load(scale, [])
        mid[tile_m, tile_n] = (x[tile_m, tile_n] + y[tile_m, tile_n]) * s

    hl.barrier()  # phase 1 below reads `mid`, written above

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = mid[tile_m, tile_n] * 2.0
    return out
```

Each `hl.barrier()`-separated phase compiles to its own private MLIR function;
the entry function calls them in order, threading tensors between phases, and
host-computed tensors are entry arguments evaluated from the host code.
`hl.barrier()` is required (not optional, not CPU-specific) whenever a later
phase reads a tensor written by an earlier one -- Helion's frontend rejects
the kernel otherwise (`LoopDependencyError`), and no statement other than
`hl.barrier()` itself may appear between two top-level device loops, so a
host tensor a later phase needs must be computed before the loop that first
uses it. See `examples/multi_phase_mlir.py` for a complete, runnable example,
and `docs/MLIR_LIMITATIONS.md` item 11 for current limitations.

## Configurable Block Sizes

Block sizes from `helion.Config(block_sizes=[...])` are propagated into generated loops.

You can inspect this using pre-lowering dumps:

```python
import io
import os
from contextlib import redirect_stdout
from unittest import mock

buf = io.StringIO()
with (
    mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
    redirect_stdout(buf),
):
    _ = add_direct(A, B)

print(buf.getvalue())
```

## Configs, Pipelines and Autotuning

A config has two keys on this backend: `block_sizes`, and `mlir_pipeline`
(`"opt"` or `"scalar"`) to pick the lighthouse pipeline for that config. Without
`mlir_pipeline`, `HELION_MLIR_PIPELINE` (`opt` or `scalar`) selects it, and `opt` is the
default. Other Helion config keys (`num_warps`, `reduction_loops`, ...) are rejected.

```python
@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32, 32], mlir_pipeline="scalar"))
```

The pipelines are `_compiler/pipeline.yaml` (`opt`: tiling, vectorization, OpenMP) and
`_compiler/scalar.yaml` (lighthouse's scalar lowering, a simpler fallback). Both begin
by lowering `linalg.pack`/`linalg.unpack` with lighthouse's `x86/pack_lowering.py`; the
opt pipeline adds its own stages for padding, runtime-shaped ops, out-of-bounds vector
transfers and LLVM legalization (`_compiler/helion_transforms.py`, see
`docs/MLIR_LIMITATIONS.md`, section 15).

Config selection follows Helion:
- One config (`config=` or `configs=[c]`) is used as is; nothing is tuned.
- Several configs (`configs=[a, b]`) are benchmarked and the fastest is used.
- Without a config, Helion's autotuner searches block sizes (`autotune_effort`
  selects the effort; `"none"` uses the default config). Candidates are timed by
  wall clock on the CPU. The best config is cached on disk (`HELION_CACHE_DIR`),
  keyed by the kernel, its inputs, the CPU model and the default pipeline;
  `HELION_FORCE_AUTOTUNE=1` re-tunes.
- A search under the optimizing pipeline tries tiles of at least 32 (where the
  dimension allows) except for the leading dim of each outermost loop, which may
  be smaller for more parallel tiles.

Compiled modules are cached in-process by their text and pipeline, so compiling
the same module again (another shape bucket with equal static sizes, a repeated
config) skips lighthouse and the JIT.

Tests and the conformance sweep set `HELION_DISALLOW_AUTOTUNING=1` and give every
kernel a fixed config.

## Inline MLIR

`helion_mlir_backend.inline_mlir(source, args, output_like, reference=...)` calls a
hand-written MLIR `func.func` (text or an `mlir.ir.Module`) on tiles inside a device
loop; it is inlined into the kernel and lowered with it. See
[INLINE_MLIR_GUIDE.md](INLINE_MLIR_GUIDE.md) and `examples/inline_mlir.py`.

## Debugging Aids

### Print generated module

```python
module = generate_mlir(add_kernel, [A, B])
print(module)
```

### Save module to file

```python
with open("kernel_ir.mlir", "w") as f:
    f.write(str(module))
```

### Useful environment variables

- `HELION_MLIR_DUMP_PRE_LOWERING=1`
  - Prints MLIR after inlining and before lighthouse lowering.

## Current Package Layout

The implementation is organized by responsibility:

```text
helion_mlir_backend/_compiler/mlir/
├── backend.py                        # backend registration, execute_mlir
├── driver.py, host_code.py           # direct-call path: host code + entry call
├── build_context.py                  # typed mutable lowering state
├── codegen.py                        # phase functions and memref entry
├── analysis/                         # geometry, tensor effects, signature, contractions
├── lowering/                         # operation-family MLIR emitters
├── aten_bridge/                      # torch-mlir helpers for generic ATen ops
└── support/                          # index resolution, types, diagnostics
```

`MLIRBackend` inherits from Helion's backend-neutral `Backend`; it does not use
the Triton Python AST code-generation path. MLIR is emitted directly as
Linalg-on-Tensors IR and lowered/JIT-compiled by `_compiler/execution.py`
(`compile_entry`).

For shape-resolution details, see `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`.

## Troubleshooting

### NoDeviceLoopsInKernel

Cause:
- Tensor operations are outside `hl.tile()` loops.

Fix:
- Move tensor ops into device loops.

### could not get source code

Cause:
- Kernel defined in interactive or dynamically generated context where source is unavailable.

Fix:
- Define the kernel in a regular Python file.

### CPU-only execution error for CUDA input

Cause:
- Current runtime path is CPU execution only.

Fix:
- Use CPU tensors for execute_mlir/direct MLIR backend execution.

## See Also

- `docs/INLINE_MLIR_GUIDE.md`
- `docs/MLIR_LIMITATIONS.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `tests/test_mlir_execution.py`
