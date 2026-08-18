# MLIR Backend Design and Architecture

## Overview

The Helion MLIR backend generates MLIR intermediate representation from high-level Helion kernel definitions. It serves as an alternative to the Triton backend for kernel compilation.

## Architecture

### High-Level View

```
Helion Kernel (Python)
        ↓
Type Propagation & Device IR (FX Graph)
        ↓
      MLIRBackend → MLIRModuleBuilder
        ↓
MLIR Module (Linalg-on-Tensors)
        ↓
Downstream Compiler (e.g., Triton, MLIR transforms)
```

### Component Breakdown

#### 1. **Entry Point: generate_mlir()**
- Location: [api.py](../helion_mlir_backend/api.py) and [backend.py](../helion_mlir_backend/_compiler/mlir/backend.py)
- Orchestrates the full compilation pipeline
- Accepts a Helion kernel and input arguments
- Returns an `mlir.ir.Module` containing the MLIR IR

**Key Steps:**
1. Extract device from tensor arguments
2. Create CompileEnvironment with MLIR backend setting
3. Convert arguments to fake tensors for type propagation
4. Run KernelCompiler to generate device IR (FX graph)
5. Instantiate MLIRModuleBuilder and generate MLIR IR
6. Return the MLIR module

#### 2. **Backend Registration: MLIRBackend**
- Location: [backend.py](../helion_mlir_backend/_compiler/mlir/backend.py)
- Registers as a compiler backend option
- Implements `generate_mlir()` method
- Inherits from Helion's backend-neutral `Backend` class, not `TritonBackend`
- Rejects Python-source-codegen-only properties because MLIR is emitted directly
- Uses the MLIR-specific `bound_kernel.py` hook for direct `backend="mlir"` calls

#### 3. **Core Lowering: MLIRModuleBuilder**
- Location: [codegen.py](../helion_mlir_backend/_compiler/mlir/codegen.py)
- Orchestrates module/function construction and dispatches to focused lowering modules
- Converts Helion's device IR (FX graph) to MLIR IR

**Architecture:**
- **State Management:**
  - `BuildContext.node_to_value`: Maps FX nodes to MLIR SSA values
  - `BuildContext.block_id_to_size`: Maps block IDs to concrete tile sizes
  - `BuildContext.block_id_to_iv`: Maps block IDs to scf loop induction variables
  - `BuildContext.param_to_value`: Maps parameters to function arguments
  - `BuildContext.forall_insert_slices`: Tracks pending parallel tensor insertions

- **Key Methods:**
  - `build()`: Entry point, creates MLIR module
  - `_build_function()`: Generates func.func with tensor signature
  - `_build_kernel_body()`: Creates scf.forall with grid-level parallelism
  - `_process_graph()`: Walks FX graph recursively
  - `_lower_node()`: Dispatches to operation-specific lowering
  - `_lower_node()`: Routes Helion and ATen nodes
  - Thin delegates: preserve the FX dispatch contract while calling modules under `lowering/`

#### 4. **Lowering Modules**

Location: [lowering/](../helion_mlir_backend/_compiler/mlir/lowering/)

- `control_flow.py`: outer `scf.forall` and nested `scf.for`
- `load_slice_ops.py`, `load_ops.py`: tile loads and gathers
- `memory_ops.py`: getitem and stores
- `matmul_ops.py`: matmul-family lowering
- `subscript_ops.py`: tensor subscripts
- `host_tensor_ops.py`: host arguments and alias materialization
- `tensor_creation_ops.py`: `full` and `zeros`
- `tile_index_ops.py`: tile-index tensor generation

#### 5. **ATen Bridge and Support**

The ATen-specific path is organized under [aten_bridge/](../helion_mlir_backend/_compiler/mlir/aten_bridge/):

- `aten_ops.py`: custom ATen registry and direct MLIR lowerings
- `aten_helper_table.py`: helper signature and identity tracking
- `helper_rebuild.py`: call-site-specific helper variants
- `torch_mlir_pipeline.py`: batched torch-mlir import and lowering

Shared utilities live under [support/](../helion_mlir_backend/_compiler/mlir/support/):

- `block_ids.py`: canonical block-key and symbolic-name parsing
- `symbolic_shape_restoration.py`: nested loop metadata repair
- `aten_prepass.py`: ATen metadata refresh
- `node_dispatch.py`, `type_utils.py`, and `errors.py`

#### 6. **Type System: torch_dtype_to_mlir()**
- Location: [type_utils.py](../helion_mlir_backend/_compiler/mlir/support/type_utils.py)
- Converts PyTorch dtypes to MLIR types
- Handles tensor shape + dtype conversion
- Supports dynamic dimensions (SymInt → `?`)

**Supported Types:**
- float16, bfloat16, float32, float64
- int8, int16, int32, int64
- uint8
- bool

## MLIR Dialect Stack

### Dialects Used

1. **func** - Function definitions
   - Wraps the entire kernel as a func.func operation
   - Signature includes tensor types (not memref)

2. **scf** - Structured Control Flow
   - **scf.forall** - Parallelizable loops over output dimensions
   - **scf.for** - Sequential loops for reductions
   - Enables implicit barrier synchronization (required by linalg)

3. **linalg** - Linear Algebra Operations
   - **linalg.matmul** - Matrix multiplication
   - **linalg.generic** - Generic element-wise operations
   - Supports implicit broadcasting

4. **tensor** - Tensor Operations
   - **tensor.extract_slice** - Extract input tiles from function arguments
   - **tensor.parallel_insert_slice** - Accumulate output tiles in parallel
   - **tensor.empty** - Create tensor placeholders

5. **arith** - Arithmetic Operations
   - **arith.constant** - Create index/float/integer constants
   - **arith.addf, arith.mulf** - Floating-point arithmetic

6. **builtin** - Fundamental types and operations
   - Tensor types: `tensor<MxNxf32>`
   - Function types

## Lowering Strategy

### Linalg-on-Tensors Philosophy

The backend generates operations on **abstract tensors**, not concrete memory buffers. This enables:

- **High-level IR**: Operations independent of memory layout
- **Optimization Opportunities**: Downstream passes can apply various transformations
- **Portability**: Same IR can target different backends (Triton, MLIR-to-LLVM, etc.)

### Key Lowering Patterns

#### 1. **Tile-Based Computation**
```
hl.tile([m, n]) → scf.forall with grid dimensions
  - Outer loops over output dimensions
  - Enable parallel execution across blocks
  - Generate index variables for slicing

hl.tile(k) → scf.for for reduction
  - Sequential inner loops
  - Accumulate partial results
```

#### 2. **Tensor Slicing**
```
out[tile_m, tile_n] = value
    ↓
tensor.extract_slice(out, offsets=[tile_m, tile_n], sizes=[...], strides=[1, 1])
    ↓
tensor.parallel_insert_slice(value, out, offsets=[...], sizes=[...])
```

#### 3. **Accumulation Pattern**
```python
acc = hl.zeros([m, n])
for k in range(...):
    acc = acc + compute()
```
Lowered to:
```
%acc = tensor.empty()
scf.forall -> {
  %partial = ...
  scf.forall.in_parallel {
    tensor.parallel_insert_slice(%partial, %acc, ...)
  }
}
```

#### 4. **Element-wise Operations**
```python
c = a + b
    ↓
linalg.generic with custom compute block:
  ^bb0(%arg_a: f32, %arg_b: f32):
    %result = arith.addf(%arg_a, %arg_b)
    linalg.yield(%result)
```

## Supported Operations

### Core Operations (Implemented)

| Operation | MLIR Mapping | Pattern |
|-----------|--------------|---------|
| Matrix Multiply | `linalg.matmul` | `C = A @ B` |
| Addition | `linalg.generic` | Element-wise add |
| Multiplication | `linalg.generic` | Element-wise mul |
| ReLU | `linalg.generic` | Element-wise max(x, 0) |
| Extract | `tensor.extract_slice` | `a[idx]` |
| Store | `tensor.parallel_insert_slice` | `out[idx] = val` |
| Constants | `arith.constant` | Tile sizes, indices |

### Not Yet Implemented

- Layer normalization
- Softmax
- Attention operations
- Dynamic reshaping
- Complex reductions (reduce_sum, etc.)
- Custom operations

## Compilation Flow Example

Given:
```python
@hl.kernel(static_shapes=True)
def matmul(x: Tensor, y: Tensor) -> Tensor:
    m, k = x.shape
    k2, n = y.shape
    out = zeros((m, n))
    for tm, tn in tile([m, n]):
        acc = zeros([tm, tn])
        for tk in tile(k):
            acc = addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out
```

Compilation steps:
1. **Parse & Type Propagation**: Extract shapes and dtypes
2. **Device IR**: Convert to FX graph with tile loops and indexing
3. **MLIR Lowering**:
   - Create func.func with tensor arguments
   - Create scf.forall over [m, n] tiles → outer loop
   - Create scf.for over [k] dimension → inner loop
   - Create linalg.matmul for each tile
   - Create tensor.parallel_insert_slice for accumulation
4. **Generate IR**: Produce valid MLIR textual representation

Generated MLIR IR:
```mlir
"builtin.module"() ({
  "func.func"() <{sym_name = "matmul", ...}> ({
  ^bb0(%arg0: tensor<MxKxf32>, %arg1: tensor<KxNxf32>):
    %out = "scf.forall"(...) ({
      %acc = "tensor.empty"() : () -> tensor<?x?xf32>
      %result = "scf.for"(...) ({
        %mm = "linalg.matmul"(...)
      })
      "scf.forall.in_parallel"({
        "tensor.parallel_insert_slice"(%mm, %out, ...)
      })
    })
    "func.return"(%out) : (tensor<MxNxf32>) -> ()
  })
})
```

## Device Abstraction

The backend treats tile dimensions (m_tile, n_tile, k_tile) as **block IDs**:

- **Block ID 0**: Output dimension M (parallelizable)
- **Block ID 1**: Output dimension N (parallelizable)
- **Block ID 2**: Reduction dimension K (sequential)

This mapping comes from Helion's device IR convention and enables proper scf.forall/scf.for placement.

## Location and Context Management

MLIR Python operations require an active `mlir.ir.Context` and `mlir.ir.Location`:

```python
ctx = ir.Context()
ctx.load_all_available_dialects()
with ctx:
    with ir.Location.unknown(ctx):
        # Create operations here
        module = ir.Module.create()
```

The backend manages this context internally in `build()`.

## Performance Considerations

1. **Tensor Abstraction Overhead**: High-level IR may have larger size than low-level code
2. **Downstream Optimization**: Performance depends on downstream compiler passes
3. **Bufferization**: Should be done by downstream pass (not in this backend)
4. **Vectorization**: Implicit in tensor operations, realized downstream

## Testing Strategy

- **Unit Tests**: Type conversions, individual operation lowerings
- **Integration Tests**: Full kernel compilation from Python to MLIR
- **IR Validity Tests**: Ensure generated MLIR parses correctly
- **Error Handling**: Invalid kernel structures properly rejected

See [tests/test_mlir_backend.py](../tests/test_mlir_backend.py) for test suite.

## Future Extensions

1. **Layer Normalization**: Add custom linalg operation or linalg.normalize
2. **Softmax**: Implement as fused reduction + exponential
3. **Attention**: Matmul-based building blocks
4. **Dynamic Shapes**: Support SymInt dimensions fully
5. **Bufferization**: Option to generate memref-based IR
6. **Custom Dialects**: Support for domain-specific operations

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [Linalg Dialect Guide](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [Helion Documentation](https://github.com/jaimicore/helion)
- [Triton Backend Design](https://triton-lang.org/)
