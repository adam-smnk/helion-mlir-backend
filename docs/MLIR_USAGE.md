# MLIR Backend Usage Guide

This guide describes the current usage of the MLIR backend in this repository.

## Current Backend State

- The backend is experimental, but it is not IR-only.
- End-to-end CPU execution is supported through the lighthouse execution path.
- Two user-facing flows are validated in tests:
  - Explicit flow: generate MLIR then execute via backend API.
  - Direct flow: call a kernel decorated with backend="mlir".

Reference tests are in `tests/test_mlir_execution.py`.

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

## Kernel Requirements

### Required

- Use `@helion.kernel(static_shapes=True)`.
- Provide tensor type annotations.
- Place tensor work inside helion device loops (`hl.tile(...)`).
- Use concrete input tensor shapes at compile time.

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

- `docs/MLIR_LIMITATIONS.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `tests/test_mlir_execution.py`
