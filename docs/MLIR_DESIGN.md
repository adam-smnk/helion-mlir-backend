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
- `driver.py` replaces `BoundKernel.compile_config` for direct `backend="mlir"` calls
  (see [Calling Convention](#calling-convention))

#### 3. **Core Lowering: MLIRModuleBuilder**
- Location: [codegen.py](../helion_mlir_backend/_compiler/mlir/codegen.py)
- Orchestrates module/function construction and dispatches to focused lowering modules
- Converts Helion's device IR (FX graph) to MLIR IR

**Architecture:**
- **State Management:**
  - `BuildContext.node_to_value`: Maps FX nodes to MLIR SSA values
  - `BuildContext.geometry`: `KernelGeometry` (block sizes, spans, loop bounds)
  - `BuildContext.block_id_to_iv`: Maps block IDs to the current tile offset
  - `BuildContext.param_to_value`: Maps parameters to function arguments
  - `BuildContext.effects` / `BuildContext.tensors`: host tensors each graph
    loads/stores (`analysis/tensor_effects.py`) and the current SSA value of every
    written host tensor (`lowering/tensor_state.py`). A store is an `insert_slice`
    into that value; a load of a written tensor reads it. Each root graph is its
    own `scf.forall` whose iterations own disjoint regions of the written tensors
    (inserted back with `parallel_insert_slice`); if a written tensor is not
    partitioned by every grid dim, the root runs as a sequential `scf.for` nest.
    Nested `scf.for` loops carry the tensors their bodies write.

- **Key Methods:**
  - `build()`: Entry point, creates the MLIR module
  - `_build_phase_function()`: one private tensor `func.func` per `hl.barrier()` phase
  - `_build_entry_function()`: the public memref-ABI entry that calls the phases
  - `_prebuild_aten_helpers()`: Batch-lowers ATen nodes without a direct lowering
  - Per-node lowering is `lowering/registry.py::lower_node`

#### 4. **Lowering Modules**

Location: [lowering/](../helion_mlir_backend/_compiler/mlir/lowering/)

- `registry.py`: `@lowers(target)` dispatch keyed by target identity (Helion API
  functions, ATen `OpOverload`s, or an `OpOverloadPacket` with an overload filter);
  each node is lowered inside its `meta["location"]` so errors name the kernel line
- `control_flow.py`: outer `scf.forall` and nested `scf.for`
- `load_slice_ops.py`, `load_ops.py`: tile loads and gathers
- `memory_ops.py`: getitem and stores
- `contraction_ops.py`: the single lowering for `mm`/`bmm`/`matmul`/`addmm`/
  `baddbmm`, `hl.dot`, captured einsum and `acc + contraction`, matched by
  `analysis/contractions.py`
- `elementwise_ops.py`: index-scalar binary ops and aliases
- `view_ops.py`, `method_ops.py`, `transpose_ops.py`: views and `Tensor` methods
- `emit.py`: shared builders (constants, fills, casts as `linalg.generic`)
- `subscript_ops.py`: tensor subscripts
- `host_tensor_ops.py`: host arguments and alias materialization
- `tensor_creation_ops.py`: `full` (also `hl.zeros`)
- `tile_index_ops.py`: tile positions, `tile.index`, shape queries

`einsum_capture.py` (at the package root) is the one piece that runs *before*
codegen: it installs a `TorchFunctionMode` around Helion's device-IR lowering
so a contractible `torch.einsum` is recorded as a single custom op instead of
being expanded by PyTorch's dispatcher. Non-contractible equations are left to
that expansion.

#### 5. **ATen Bridge and Support**

The ATen-specific path is organized under [aten_bridge/](../helion_mlir_backend/_compiler/mlir/aten_bridge/):

- `helper_call.py`: call-site `func.call` to a helper
- `aten_helper_table.py`: helper signature and identity tracking
- `helper_rebuild.py`: call-site-specific helper variants
- `torch_mlir_pipeline.py`: batched torch-mlir import and lowering

Shared utilities live under [support/](../helion_mlir_backend/_compiler/mlir/support/):

- `block_ids.py`: canonical block-key and symbolic-name parsing
- `symbolic_shape_restoration.py`: nested loop metadata repair
- `aten_prepass.py`: ATen metadata refresh
- `einsum_spec.py`: einsum equation analysis against `linalg.contract` semantics
- `type_utils.py` and `errors.py` (errors are `helion.exc.BaseError`s)

#### 6. **Type System: torch_dtype_to_mlir()**
- Location: [type_utils.py](../helion_mlir_backend/_compiler/mlir/support/type_utils.py)
- Converts PyTorch dtypes to MLIR types
- Handles tensor shape + dtype conversion
- Supports dynamic dimensions (SymInt → `?`)

**Supported Types:**
- float16, bfloat16, float32, float64
- int8, int16, int32, int64
- bool (uint8 is rejected: integers are lowered with signed semantics)

## MLIR Dialect Stack

### Calling Convention

One module per compiled config holds the whole kernel:

- `@<kernel>__phase<i>(ins..., inouts..., scalars...) -> (inouts...)`: private,
  pure tensor functions, one per `hl.barrier()`-separated phase.
- `@<kernel>(inouts..., ins..., scalars...)`: the public entry, over memrefs. It
  wraps inputs with `bufferization.to_tensor ... restrict` and inouts with
  `bufferization.to_tensor ... restrict writable`, calls the phases in order
  (threading SSA values; a barrier is just the call boundary) and commits every
  final inout with `bufferization.materialize_in_destination ... restrict
  writable`. Runtime scalars are 0-d memrefs (`f64` for floats, `i64` for ints).

The arguments are the host tensors the device code uses
([analysis/signature.py](../helion_mlir_backend/_compiler/mlir/analysis/signature.py)):
declared tensor parameters in declaration order, then other host tensors (locals,
globals) in first-use order. A tensor written anywhere is an inout, everything
else an input. A read-only host view of a declared parameter is not an argument:
it lowers to a reshape of that parameter. Each entry argument carries
`{helion.name, helion.role, helion.param}` attributes: the host expression that
produces it, `in`/`inout`/`scalar`, and its position among the tensor
parameters.

`restrict` means no two arguments may alias; `writable` means the kernel may
write the buffer in place. Because inouts start from the caller's tensor, host
initialization (`torch.full_like`), in-place updates and `out=` parameters keep
their Helion semantics.

Lighthouse's `result_to_args` is not used (`BackendDriver(result_to_args=False)`):
it only turns results into fresh output buffers, marks every input `restrict`
(so one buffer cannot be both input and output), rejects returning an argument
unchanged, and is tensor-only. Runner, `TorchMemoryManager`, pipelines and
schedules stay lighthouse's.

**Call path** ([driver.py](../helion_mlir_backend/_compiler/mlir/driver.py),
[host_code.py](../helion_mlir_backend/_compiler/mlir/host_code.py)). Every call
runs the kernel's host code up to the device loops, evaluates each entry
argument's host expression in the host locals, calls the entry, then runs the
host code after the loops and returns the kernel's own `return` value. Host-side
block sizes (`hl.register_block_size`) become the config's values, as in
Helion's host codegen. Tensors are passed contiguous: free when they already
are, otherwise one copy in plus a copy back for inouts (strided memrefs are
untested in the lighthouse pipelines). An input that overlaps an inout is cloned
so `restrict` holds (it reads the value from before the call); two inouts that
share memory are rejected.

**Entry points.** Direct calls (`@helion.kernel(backend="mlir")`) and
`compile_mlir(kernel, args)` use the call path above. `generate_mlir` returns the
module for inspection. `MLIRBackend.execute_mlir(module, *tensor_params)` runs no
host code: it zero-initializes host-created inouts, rejects kernels that read
host-computed tensors or take runtime scalars, and returns every inout.

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
| Einsum (2 operands) | `linalg.contract` | `torch.einsum("mk,kn->mn", a, b)` |
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
   - Create the phase `func.func` with tensor arguments (`x`, `y` in, `out` inout)
   - Create scf.forall over [m, n] tiles → outer loop, `shared_outs` = `out`
   - Create scf.for over [k] dimension → inner loop
   - Create linalg.matmul for each tile
   - Create tensor.parallel_insert_slice of each iteration's owned `out` tile
   - Create the memref entry that calls the phase
4. **Generate IR**: Produce valid MLIR textual representation

Generated MLIR IR (abbreviated):
```mlir
func.func private @matmul__phase0(%x: tensor<MxKxf32>, %y: tensor<KxNxf32>,
                                  %out: tensor<MxNxf32>) -> tensor<MxNxf32> {
  %r = scf.forall (%i, %j) in (M/bm, N/bn) shared_outs(%o = %out) -> (tensor<MxNxf32>) {
    %acc = linalg.fill ... -> tensor<bmxbnxf32>
    %mm = scf.for %k = ... iter_args(%a = %acc) -> (tensor<bmxbnxf32>) {
      %t = linalg.matmul ins(...) outs(%a : tensor<bmxbnxf32>) -> tensor<bmxbnxf32>
      scf.yield %t : tensor<bmxbnxf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %mm into %o[...] ...
    }
  }
  return %r : tensor<MxNxf32>
}
func.func public @matmul(%out: memref<MxNxf32>, %x: memref<MxKxf32>, %y: memref<KxNxf32>) {
  %o = bufferization.to_tensor %out restrict writable : ...
  %xt = bufferization.to_tensor %x restrict : ...
  %yt = bufferization.to_tensor %y restrict : ...
  %r = call @matmul__phase0(%xt, %yt, %o) : ...
  bufferization.materialize_in_destination %r in restrict writable %out : ...
  return
}
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
3. **Bufferization**: Done by lighthouse's pipelines; the backend only fixes the
   function boundary (`to_tensor` / `materialize_in_destination` in the entry)
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
