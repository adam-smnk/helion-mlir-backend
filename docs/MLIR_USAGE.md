# MLIR Backend Usage Guide

## Quick Start

### Installation

The MLIR backend is included with Helion when the `mlir-python-bindings` package is installed:

```bash
pip install helion mlir-python-bindings
```

### Basic Usage

```python
import torch
import helion as hl
from helion.mlir import generate_mlir

# Define a kernel
@hl.kernel(static_shapes=True)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    k2, n = y.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out

# Generate MLIR IR
device = torch.device("cpu")
x = torch.randn(64, 128, dtype=torch.float32, device=device)
y = torch.randn(128, 96, dtype=torch.float32, device=device)

module = generate_mlir(matmul_kernel, [x, y])

# Print the generated MLIR
print(module)
```

## API Reference

### generate_mlir()

**Signature:**
```python
def generate_mlir(
    kernel: Kernel,
    args: List[torch.Tensor],
    *,
    config: Optional[Config] = None
) -> mlir.ir.Module:
```

**Parameters:**
- `kernel`: A `@hl.kernel`-decorated function
- `args`: List of input tensors (must be concrete tensors, not symbolic)
- `config` (optional): Compilation configuration (default: kernel's default_config)

**Returns:**
- `mlir.ir.Module`: MLIR intermediate representation

**Raises:**
- `ValueError`: If kernel is not properly decorated
- `NoDeviceLoopsInKernel`: If kernel lacks required `hl.tile()` loops
- `CompilationError`: If compilation fails

**Example:**
```python
module = generate_mlir(my_kernel, [x, y])
ir_string = str(module)
```

### MLIRBackend

**Class:** `helion._compiler.mlir.backend.MLIRBackend`

Implements the backend interface for Helion compilation pipeline.

**Methods:**
- `generate_mlir(host_function, config, env) -> ir.Module`
  - Internal method called by `generate_mlir()` function
  - Not typically called directly

**Registration:**
```python
from helion._compiler.backend_registry import get_backend_class
backend = get_backend_class("mlir")()
```

## Kernel Structure Requirements

### Mandatory Elements

1. **@hl.kernel Decorator**
```python
@hl.kernel(static_shapes=True)  # static_shapes=True is required
def kernel(...): ...
```

2. **Type Annotations**
```python
def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
```

3. **Tile Loops**
All tensor operations must be inside `hl.tile()` loops:
```python
for tile_m, tile_n in hl.tile([m, n]):
    # Operations here are tiled
    result = operation(x[tile_m, tile_n])
```

4. **Static Shapes**
Dimensions must be known at compile time:
```python
m, n = x.shape  # Must work with concrete shapes
# NOT: m = x.shape[0] if condition else ...
```

### Nested Tiling Pattern

For operations with reduction dimensions (like matmul):

```python
@hl.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    k2, n = y.shape

    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    # Outer loop: output dimensions (parallelizable)
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulator for this tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        # Inner loop: reduction dimension (sequential)
        for tile_k in hl.tile(k):
            # Accumulation
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

        # Store result
        out[tile_m, tile_n] = acc

    return out
```

## Common Patterns

### Pattern 1: Simple Element-wise

```python
@hl.kernel(static_shapes=True)
def relu(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])

    return out
```

### Pattern 2: Binary Element-wise

```python
@hl.kernel(static_shapes=True)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

    return out
```

### Pattern 3: Matrix Multiplication

```python
@hl.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

### Pattern 4: Fused Operations

```python
@hl.kernel(static_shapes=True)
def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    k2, n = y.shape
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])

        # Fuse ReLU with matmul result
        out[tile_m, tile_n] = torch.relu(acc)

    return out
```

## Supported Data Types

| PyTorch Type | MLIR Type |
|--------------|-----------|
| torch.float16 | f16 |
| torch.bfloat16 | bf16 |
| torch.float32 | f32 |
| torch.float64 | f64 |
| torch.int8 | i8 |
| torch.int16 | i16 |
| torch.int32 | i32 |
| torch.int64 | i64 |
| torch.uint8 | ui8 |
| torch.bool | i1 |

Example:
```python
# float32
x = torch.randn(64, 64, dtype=torch.float32)

# float64
y = torch.randn(64, 64, dtype=torch.float64)

# int32
z = torch.randint(0, 100, (64, 64), dtype=torch.int32)
```

## Debugging and Inspection

### Print Generated MLIR

```python
module = generate_mlir(kernel, [x, y])
ir_string = str(module)
print(ir_string)
```

### Check IR Components

```python
ir_str = str(module)

# Check for operations
print(f"Has func: {'func.func' in ir_str}")
print(f"Has forall: {'scf.forall' in ir_str}")
print(f"Has matmul: {'linalg.matmul' in ir_str}")
print(f"Has generic: {'linalg.generic' in ir_str}")
```

### Save IR to File

```python
with open("kernel_ir.mlir", "w") as f:
    f.write(str(module))
```

### Verify IR Syntax

```python
import mlir.ir as ir

try:
    ir.Module.parse(ir_string)
    print("IR is syntactically valid")
except Exception as e:
    print(f"IR parsing failed: {e}")
```

## Environment Variables and Configuration

### Backend Selection

To use MLIR backend instead of default Triton:

```python
# Option 1: Direct API
from helion.mlir import generate_mlir
module = generate_mlir(kernel, args)

# Option 2: Via compilation environment
from helion import CompileEnvironment
env = CompileEnvironment(device, Settings(backend="mlir"))
```

### Python Bindings Version

Required version: `mlir-python-bindings` from EUDSL index

```bash
pip show mlir-python-bindings
```

## Performance Notes

1. **IR Size**: MLIR IR is high-level and may be larger than optimized code
2. **Downstream Compilation**: Performance depends on downstream passes (Triton, MLIR-to-LLVM)
3. **No Direct Execution**: This backend generates IR only, not runnable code
4. **Optimization Pass**: Recommend applying MLIR optimization passes on generated IR

## Troubleshooting

### Error: "The variable name X is reserved"

**Cause**: Kernel uses reserved names (like `hl`, `tile`, etc.)

**Solution**: Avoid reserved names:
```python
# BAD
import helion as hl
@hl.kernel
def kernel():
    for hl in range(10): ...  # hl is reserved

# GOOD
import helion
@helion.kernel
def kernel():
    ...
```

### Error: "NoDeviceLoopsInKernel"

**Cause**: Tensor operations outside tile loops

**Solution**: Wrap all tensor operations in `hl.tile()`:
```python
# BAD
out = x + y

# GOOD
for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
```

### Error: "An MLIR function requires a Context"

**Cause**: MLIR operations created outside active context

**Solution**: This should not occur with `generate_mlir()`. If it does, file an issue.

### Error: "could not get source code"

**Cause**: Kernel defined in REPL or dynamically without source file

**Solution**: Define kernel in a Python file or use `@hl.kernel` decorator properly

## Examples

See [examples/](../examples/) directory for complete working examples:
- [matmul_mlir.py](../examples/matmul_mlir.py) - Matrix multiplication
- [elementwise_mlir.py](../examples/elementwise_mlir.py) - Element-wise operations
- [fused_ops_mlir.py](../examples/fused_ops_mlir.py) - Fused operations

Run examples:
```bash
python examples/matmul_mlir.py
python examples/elementwise_mlir.py
python examples/fused_ops_mlir.py
```
