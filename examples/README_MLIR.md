# MLIR Backend Examples

This directory contains example kernels that demonstrate how to use the Helion MLIR backend to generate MLIR IR from high-level kernel definitions.

## Overview

The MLIR backend enables Helion kernels to be lowered into MLIR code using the Linalg-on-Tensors abstraction. These examples show:

- Basic kernel structure with tile-based loops
- How the backend generates MLIR operations
- Various kernel patterns: matmul, element-wise, fused operations
- MLIR IR inspection and debugging

## Examples

### 1. matmul_mlir.py
**Tiled Matrix Multiplication**

Demonstrates a basic tiled matmul kernel `C = A @ B` with:
- Outer loop over output tile dimensions (M, N)
- Inner loop over reduction dimension (K)
- `linalg.matmul` operations for each tile
- `tensor.extract_slice` for input tiling
- `tensor.parallel_insert_slice` for output accumulation

**Run:**
```bash
python examples/matmul_mlir.py
```

**Output:** Displays generated MLIR IR with:
- `scf.forall` for parallelizable outer loops
- `scf.for` for sequential reduction loop
- `linalg.matmul` for matrix multiplication
- Tensor slicing and insertion operations

---

### 2. elementwise_mlir.py
**Element-wise Operations (Add, Multiply, ReLU)**

Demonstrates basic element-wise operations:
- **Addition:** `C = A + B`
- **Multiplication:** `C = A * scale`
- **ReLU:** `C = max(A, 0)`

Each uses `hl.tile()` loops and `linalg.generic` for flexible element-wise computation.

**Run:**
```bash
python examples/elementwise_mlir.py
```

**Output:** Summary of MLIR IR generation for each operation showing:
- Loop structure (scf.forall over tiles)
- Operation kind (linalg.generic for custom operations)
- Generated IR size and key components

---

### 3. fused_ops_mlir.py
**Fused Operations (Matmul + ReLU, Matmul + Bias)**

Demonstrates kernel fusion patterns:
- **Matmul + ReLU:** `C = max(A @ B, 0)` - shows operation composition
- **Matmul + Bias:** `C = A @ B + bias` - shows bias broadcasting with indexing

Useful for understanding how downstream MLIR passes can fuse and optimize.

**Run:**
```bash
python examples/fused_ops_mlir.py
```

**Output:** Detailed IR analysis for each fused operation.

---

## Running the Examples

### Prerequisites
- Helion installed with MLIR backend support
- MLIR Python bindings available
- PyTorch installed

### Run All Examples
```bash
cd /home/asiemien/helion-mlir
python examples/matmul_mlir.py
python examples/elementwise_mlir.py
python examples/fused_ops_mlir.py
```

### Run Specific Example
```bash
python examples/matmul_mlir.py
```

### Inspect Generated MLIR
Each example prints the full MLIR IR module. You can capture and analyze it:

```bash
python examples/matmul_mlir.py > matmul_ir.mlir
cat matmul_ir.mlir
```

## Understanding the MLIR Output

Generated MLIR modules use three primary dialects:

1. **func** - Function definitions and calls
2. **scf** - Structured control flow (forall, for, while)
3. **linalg** - High-level tensor operations (matmul, generic)
4. **tensor** - Tensor operations (extract_slice, insert_slice)
5. **arith** - Arithmetic operations (constants, index math)

### Example IR Pattern

```mlir
"builtin.module"() ({
  "func.func"() <{
    sym_name = "kernel_name",
    function_type = (tensor<MxKxf32>, tensor<KxNxf32>) -> tensor<MxNxf32>
  }> ({
  ^bb0(%arg0: tensor<MxKxf32>, %arg1: tensor<KxNxf32>):
    # scf.forall over output tiles (M, N)
    %0 = "scf.forall"(...) ({
      # Inner loops and computations
      %1 = "linalg.matmul"(...)
      # tensor.parallel_insert_slice for accumulation
      "scf.forall.in_parallel"({
        "tensor.parallel_insert_slice"(...)
      })
    })
    "func.return"(%0)
  })
})
```

## Key Helion Patterns

### Tile Loop Requirement
All tensor operations **must** be inside `hl.tile()` loops:

```python
@hl.kernel(static_shapes=True)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros((m, n), device=x.device)
    
    for tile_m, tile_n in hl.tile([m, n]):
        # Operations here are tiled
        out[tile_m, tile_n] = operation(x[tile_m, tile_n])
    
    return out
```

### Nested Tiling
Reduction dimensions require nested tile loops:

```python
for tile_m, tile_n in hl.tile([m, n]):
    acc = hl.zeros([tile_m, tile_n])
    for tile_k in hl.tile(k):  # Nested for reduction
        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
    out[tile_m, tile_n] = acc
```

### Static Shapes Only
Current MLIR backend supports static shapes:

```python
@hl.kernel(static_shapes=True)  # Required
def kernel(x: torch.Tensor) -> torch.Tensor:
    # Shape dimensions must be known at compile time
    m, n = x.shape
    ...
```

## Limitations

1. **Static shapes only** - Dynamic tensor dimensions not supported
2. **Supported operations** - See [LIMITATIONS.md](../docs/LIMITATIONS.md)
3. **No dynamic buffers** - Memory layout must be predetermined
4. **Scalar arguments** - Limited support for non-tensor arguments

## Next Steps

- See [DESIGN.md](../docs/DESIGN.md) for architecture details
- Check [USAGE.md](../docs/USAGE.md) for the `generate_mlir()` API
- Review [LIMITATIONS.md](../docs/LIMITATIONS.md) for constraints
- Examine test cases in [tests/test_mlir_backend.py](../tests/test_mlir_backend.py)

## Debugging

### Print Full IR
```python
from helion.mlir import generate_mlir
module = generate_mlir(kernel, args)
print(module)  # Full MLIR IR
```

### Check IR Dialects
```python
ir_str = str(module)
print("Dialects used:")
print(f"  func: {'func' in ir_str}")
print(f"  scf: {'scf' in ir_str}")
print(f"  linalg: {'linalg' in ir_str}")
print(f"  tensor: {'tensor' in ir_str}")
```

### Run Tests
```bash
cd /home/asiemien/helion-mlir
python -m pytest tests/test_mlir_backend.py -v
```
