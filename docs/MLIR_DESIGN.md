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
  - `BuildContext.sizes`: static or `index` values of sizes (`sizes.py`, see
    Runtime sizes)
  - `BuildContext.effects` / `BuildContext.tensors`: host tensors each graph
    loads/stores (`analysis/tensor_effects.py`) and the current SSA value of every
    written host tensor (`lowering/tensor_state.py`). A store is an `insert_slice`
    into that value; a load of a written tensor reads it. Each root graph is its
    own `scf.forall` whose iterations own disjoint regions of the written tensors
    (inserted back with `parallel_insert_slice`); if a written tensor is not
    partitioned by every grid dim, the root runs as a sequential `scf.for` nest.
    A store index partitions a grid dim if it is the dim's tile, or a scalar equal
    to its offset (`hl.grid` index, `tile.begin`) or its tile number (`tile.id`),
    or the tile of a nested loop over exactly one tile of that dim
    (`hl.tile(t.begin, t.end)`, recorded in `KernelGeometry.enclosing_tiles`): such
    a tile lies inside the iteration's region and is indexed from its origin. A
    nested loop over a full enclosing tile whose block size divides it has only
    full tiles, so its slices are static.
    Nested `scf.for` loops carry the tensors their bodies write.

- **Key Methods:**
  - `build()`: Entry point, creates the MLIR module
  - `_build_phase_function()`: one private tensor `func.func` per `hl.barrier()` phase
  - `_build_entry_function()`: the public memref-ABI entry that calls the phases
  - `AtenHelperTable.materialize()`: after all functions are built, lowers the
    requested ATen helpers (one torch-mlir run) and clones them into the module
  - Per-node lowering is `lowering/registry.py::lower_node`

#### 4. **Lowering Modules**

Location: [lowering/](../helion_mlir_backend/_compiler/mlir/lowering/)

- `registry.py`: `@lowers(target)` dispatch keyed by target identity (Helion API
  functions, ATen `OpOverload`s, or an `OpOverloadPacket` with an overload filter);
  each node is lowered inside its `meta["location"]` so errors name the kernel line
- `loops.py`: one `scf.forall` (or sequential `scf.for` nest) per root graph and
  one `scf.for` per block id of a nested `_for_loop`, carrying the variables the
  body assigns and the tensors it writes
- `control_flow.py`: `scf.if` for `_if` and `scf.while` for `_while_loop`, each
  carrying the tensors its body writes; `_phi`/`_new_var`; the subgraph and
  loop-carried value helpers shared with `loops.py` (loop outputs are matched
  to their variables through `_phi(before, getitem(loop, i))`)
- `scalar_ops.py`: arithmetic, comparisons and `_and`/`_or`/`_not` of scalar
  (`SymInt`) values, e.g. `if` conditions
- `combine_ops.py`: `hl.reduce` and `hl.associative_scan`/`torch.cumsum` with a
  combine function, as a sequential `scf.for`; a reduction whose combine function
  is one `add`/`mul`/`maximum`/`minimum`/`logical_and`/`logical_or` of its two
  arguments is a `linalg.reduce` instead
- `unsupported_ops.py`: Helion operations rejected with their reason (atomics,
  random numbers, inline code, `device_print`)
- `load_slice_ops.py`: tile loads; a 1-D index tensor in one dimension gathers
  through the `aten.index.Tensor` helper
- `memory_ops.py`: getitem, `_mask_to` and stores (tensor-indexed stores are
  rejected; a size-1 dim of the value is broadcast into the destination)
- `contraction_ops.py`: the single lowering for `mm`/`bmm`/`matmul`/`addmm`/
  `baddbmm`, `hl.dot`, captured einsum and `acc + contraction`, matched by
  `analysis/contractions.py`
- `elementwise_ops.py`: index-scalar binary ops, aliases, Helion's GELU ops
- `view_ops.py`, `transpose_ops.py`: views, `hl.subscript`
  (new axes: Helion admits only `None` and `:`), `hl.split`/`hl.join` and
  `permute` (Helion traces every transpose and tensor method as ATen ops)
- `emit.py`: shared builders (constants, fills, casts as `linalg.generic`)
- `host_tensor_ops.py`: host arguments and reshapes of the parameters they alias
- `inline_mlir_ops.py`: `inline_mlir` calls to the user's function, cloned into
  the module by `snippets.py`, with operands and results cast to its declared types
- `tensor_creation_ops.py`: `full` (also `hl.zeros`) and `torch.tensor` constants
- `tile_index_ops.py`: tile positions, `tile.index`, shape queries

`trace_mode.py` (at the package root) is the one piece that runs *before*
codegen: it installs a `TorchFunctionMode` around Helion's device-IR lowering
so a contractible `torch.einsum` is recorded as a single custom op instead of
being expanded by PyTorch's dispatcher. Non-contractible equations are left to
that expansion. The mode also routes tensor methods that mirror a `torch`
function with a Helion device replacement (`x.cumsum(d)`, `x.cumprod(d)`) to
that replacement; Helion replaces only the function form.

`inline_mlir` ([language.py](../helion_mlir_backend/language.py), guide in
[INLINE_MLIR_GUIDE.md](INLINE_MLIR_GUIDE.md)) is a plain function: under the MLIR
backend it calls the device-only API op `_inline_mlir`, and elsewhere (other
backends, Helion's ref mode) it calls the user's `reference`, which Helion then
traces like any device code. The op's fake implementation parses and checks the
snippet against the call while Helion traces the kernel
([snippets.py](../helion_mlir_backend/_compiler/mlir/snippets.py)); the lowering
clones its functions into the module, private and renamed per snippet, and calls
the entry, so the inliner merges it into the kernel before bufferization. An
`mlir.ir.Module` global reaches it as text: `inject.py` extends Helion's
`CompileEnvironment.to_fake`, which accepts only known global types.

#### 5. **ATen Bridge and Support**

[aten_bridge/](../helion_mlir_backend/_compiler/mlir/aten_bridge/) lowers every ATen
node without a direct lowering as a `func.call` to a private helper function typed
at the call site (`helpers.py`: `call_helper`, `infer_results`, `AtenHelperTable`):

- The operands are the node's inputs as they were lowered: tensors with their
  MLIR types, and runtime scalars (kernel `float`/`int` parameters, tile
  positions) as `f64`/`i64`/`i1`. Constant scalars and non-tensor arguments are
  literals. Result types come from running the op on samples of the operand
  types (`samples.py`), so nothing is derived from Helion's symbolic metadata:
  meta tensors, or fake tensors and fresh size symbols for `?` dims and for
  runtime scalars at `SymInt` arguments. Each module has its own fake tensor
  mode (`AtenHelperTable.sampler`), freed with the module.
- The helper's name hashes the target, literals and operand types; one helper
  serves every call site with the same signature.
- Inputs that Helion's `strip_unused_inputs` masked as `None` (`x * x` becomes
  `mul(x, None)`) are restored from the arguments recorded just before it runs
  (`original_args.py`, installed by `inject.py`).
- After the module is built, the helpers missing from the process-wide
  `helper_cache.CACHE` are imported with torch-mlir's `FxImporter` and lowered to
  Linalg in one run. If that run fails, each helper is lowered alone and the first
  failing one raises an `UnsupportedOperationError` at its node's kernel source
  line.
- `infer_results` gives the same meta results to direct lowerings that need a
  result shape (`view`/`reshape`); `call_helper` builds a helper call for an op
  that is not the node's own target (gathers use `aten.index.Tensor`, Helion's
  `_gelu_erf` uses `aten.gelu`).

Helion's node metadata is read, never modified.

Shared utilities live under [support/](../helion_mlir_backend/_compiler/mlir/support/):

- `block_ids.py`: canonical block-key and symbolic-name parsing
- `index_meta.py`: index expressions to block ids
- `einsum_spec.py`: einsum equation analysis against `linalg.contract` semantics
- `type_utils.py` and `errors.py` (errors are `helion.exc.BaseError`s)

#### 6. **Type System: torch_dtype_to_mlir()**
- Location: [type_utils.py](../helion_mlir_backend/_compiler/mlir/support/type_utils.py)
- Converts PyTorch dtypes to MLIR types
- `static_dim`: a symbolic tensor dim (`SymInt`) as MLIR `?`

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
else an input. A read-only host reshape of a declared parameter (the same
elements in the same order) is not an argument: it lowers to a reshape of that
parameter; any other host view is an input of its own. Each entry argument carries
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

When a loop or tensor ends inside a tile, the slice size is its real part,
`affine.min(tile, end - offset)`: a load extracts that part and `tensor.pad`s it
with zeros to the static tile, `_mask_to` selects the reduction identity past
it, and a store extracts it from the value before inserting. Divisible extents
keep static sizes.

#### Runtime sizes

With `static_shapes=False`, a size with free symbols after substituting the
config's block sizes is `?` in the types. `ctx.sizes.value(expr)` (`mlir/sizes.py`)
turns it into
an `index` value: a size symbol is `tensor.dim` of the first host tensor
argument with that size, or the runtime scalar that carries it; compound
expressions (`s0 // 2`) are `index` arithmetic. The values are emitted at the
start of the phase function and shared by equal expressions, so a loop over a
tensor's extent and that tensor's slices use one value and need no extra clamp.
No size is ever evaluated to its example value. Forall trip counts are
`ceildiv` of the runtime span; a full slice of a runtime dim is a runtime-sized
tile; ATen helpers get fake operands with one fresh size symbol per `?` dim.

#### 3. **Accumulation Pattern**
```python
acc = hl.zeros([m, n])
for k in range(...):
    acc = acc + compute()
```
Lowered to:
```
%init = linalg.fill ...
%acc = scf.for ... iter_args(%a = %init) {
  %next = linalg.generic ins(%partial) outs(%a) { addf(%out, %in) }
  scf.yield %next
}
```
After inlining, `execution.inline_module` runs `mlir/in_place.py`: an
elementwise update of a loop-carried value that reads it takes the iter arg as
its destination (as `linalg.matmul outs(acc)` already does), and
`acc = acc + x.sum(-1)` (also `*`, `max`, `min`) becomes the reduction started
from `acc`. Bufferization then needs no buffer per iteration for them.

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
| Other ATen ops | torch-mlir helper (`linalg.*`) | reductions, softmax, norms, activations |

### Not Yet Implemented

- Scatter stores and tuple inputs of `hl.reduce`/`hl.associative_scan`
- `if` on a tensor of more than one element
- Atomics, `hl.rand`, inline assembly/Triton and `device_print` (rejected with
  the reason)

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
3. **Bufferization**: Done by lighthouse's pipelines; the backend fixes the
   function boundary (`to_tensor` / `materialize_in_destination` in the entry)
   and makes loop-carried updates destination-passing (`mlir/in_place.py`)
4. **Vectorization**: Implicit in tensor operations, realized downstream
5. **Compilation and tuning**: `execution.compile_entry` caches compiled entries
   by module text and pipeline. `MLIRBackend` keeps Helion's autotune semantics
   (block sizes only, CPU wall-clock timing, `mlir/autotune.py`'s CPU-keyed
   best-config cache); the `mlir_pipeline` config key picks the pipeline

## Testing Strategy

- **Unit Tests**: Type conversions, individual operation lowerings
- **Integration Tests**: Full kernel compilation from Python to MLIR
- **IR Validity Tests**: Ensure generated MLIR parses correctly
- **Error Handling**: Invalid kernel structures properly rejected

See [tests/test_mlir_backend.py](../tests/test_mlir_backend.py) for test suite.

## Future Extensions

Layer normalization, softmax and attention building blocks already lower through
the ATen bridge (torch-mlir helper functions inlined into the kernel).

1. **Bufferization**: Option to emit memref-based IR directly
2. **Custom Dialects**: Support for domain-specific operations

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [Linalg Dialect Guide](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [Helion Documentation](https://github.com/jaimicore/helion)
- [Triton Backend Design](https://triton-lang.org/)
