# Helion Compilation Flow: Executive Summary

## Project Overview

**Helion** is a compiler infrastructure for kernel DSLs that lowers Python/PyTorch-style kernels into multiple backend IRs:
- **Triton**: NVIDIA CUDA via Triton compiler
- **Pallas**: Google TPU via JAX Pallas/Mosaic
- **CuTe**: NVIDIA CUDA via CuTe (layout algebra)
- **TileIR**: Intermediate IR/fallback
- **Metal**: Apple Metal (experimental)

The key innovation is **backend-agnostic lowering**: Helion compiles user kernels into a common Device IR (FX graphs), then each backend generates its own code from that IR.

## MLIR Backend Reading Guide

If you are working on the MLIR backend specifically, start with these docs:

- `docs/MLIR_USAGE.md` for current execution workflow and supported kernel patterns.
- `docs/MLIR_LIMITATIONS.md` for current constraints and known boundaries.
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md` for shape resolution details across codegen and ATen helper preprocessing.

## Compilation Pipeline: 10-Stage Journey

```
┌─────────────────────────────────────────────────────────────────┐
│ User-Written Kernel (Python + PyTorch ops)                     │
│   @helion.kernel()                                              │
│   def my_kernel(x, y, ...): ...                                │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. KERNEL DECORATION                                            │
│ - @helion.kernel() wraps function                              │
│ - Captures: configs, settings, autotuning hints                │
│ Output: Kernel object with metadata                            │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. PARSING (KernelCompiler.parse)                              │
│ - Extract source code via inspect.getsource()                  │
│ - Parse to Python AST                                          │
│ - Convert to ExtendedAST (preserves source locations)          │
│ - Validate decorator ordering                                  │
│ Output: HostFunction + KernelDefinition                        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. STATIC LOOP UNROLLING                                        │
│ - Unroll hl.static_for() loops into straight-line code         │
│ Output: Simplified AST                                         │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. BACKEND-SPECIFIC AST CUSTOMIZATION                          │
│ - Backend.customize_ast() hook                                 │
│ - Rewrite high-level patterns for backend optimization         │
│ Output: Optimized AST                                          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. TYPE PROPAGATION                                             │
│ - Infer types for all variables (TensorType, NumericType, etc.)│
│ - Validate type consistency                                    │
│ Output: TypeInfo annotations on AST nodes                      │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. CONFIG FINALIZATION                                          │
│ - Prepare autotuning configuration options                     │
│ - Set up search space constraints                              │
│ Output: ConfigSpec ready for autotuning                        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 7. DEVICE IR LOWERING (WalkHostAST)          [BACKEND-AGNOSTIC] │
│ - Walk host AST structure (for loops, if statements)           │
│ - Extract device code into separate FX (functional) graphs     │
│ - Analyze memory operations, reductions, dependencies          │
│ Output: DeviceIR containing FX graphs                          │
│   - RootGraphInfo: top-level kernel code                       │
│   - ForLoopGraphInfo: device loop bodies                       │
│   - IfGraphInfo: conditional branches                          │
└───────────────────┬──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 8. INDUCTOR LOWERING (prepare_graph_lowerings)  [BACKEND ENTRY] │
│ - Walk FX graph nodes                                          │
│ - Attach backend-specific lowering objects:                    │
│   * aten_lowering_dispatch: ATen ops → backend IR              │
│   * APIFuncLowering: Helion language ops                       │
│ - Inductor's GraphLowering creates TensorBox/StorageBox IR     │
│ Output: FX nodes annotated with lowering metadata              │
│   KEY POINT: Becomes backend-specific here                     │
└───────────────────┬──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 9. CODE GENERATION (GenerateAST)                                 │
│ - Walk DeviceIR graphs in order                                │
│ - Call codegen() on each node's lowering object                │
│ - Lowering objects emit AST statements:                        │
│   * Block/grid indexing                                        │
│   * Tile computation                                           │
│   * Load/store operations (with indexing strategies)           │
│   * Reductions, atomics, barriers                              │
│   * Device-side loops                                          │
│ - Each backend's codegen module provides implementations:       │
│   * triton/memory_ops.py, triton/reduce_ops.py, etc.          │
│   * pallas/memory_ops.py, pallas/reduce_ops.py, etc.          │
│   * Registered via @_decorators.codegen(op, "<backend>")       │
│ Output: Backend-specific Python AST                            │
└───────────────────┬──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 10. SOURCE EMISSION & DECORATION                                 │
│ - Convert AST → Python source code string                      │
│ - Wrap with backend decorator:                                 │
│   * Triton: @triton.jit                                        │
│   * Pallas: plain function                                     │
│   * CuTe: custom wrapper                                       │
│ - Add imports: tl.*, hl.*, torch, etc.                         │
│ Output: Executable kernel code ready for dispatch              │
└──────────────────────────────────────────────────────────────────┘
```

## Key Architectural Insights

### 1. **Three-Layer Abstraction**
- **Layer 1 (Frontend)**: User kernel AST → Device IR (backend-agnostic)
- **Layer 2 (Lowering)**: Device IR → Inductor IR + Lowering objects (backend-specific)
- **Layer 3 (Codegen)**: Lowering objects → Backend-specific AST → Source code

### 2. **Backend Abstraction via Lowering Objects**
Each operation gets a backend-specific "lowering" object attached during Inductor lowering:
```
FX Graph Node → Lowering object (e.g., TritonMemoryOp, PallasMemoryOp)
                ↓
                Lowering.codegen() → Backend-specific AST
```

This design decouples op semantics from implementation details.

### 3. **FX Graphs as Device IR**
Instead of custom IR dataclasses (like Triton's own IR), Helion uses PyTorch's FX graphs:
- **Advantage**: Reuses PyTorch's graph infrastructure, FX pass ecosystem
- **Representation**: Nodes with `meta["lowering"]` dictionaries carrying backend info
- **Traceability**: Source locations preserved via `meta["location"]`

### 4. **Tile-Based Iteration Model**
Core abstraction for kernel blocking/scheduling:
- User writes: `for tile_m, tile_n in hl.tile([m, n]):`
- Compiler generates different loop structures per backend:
  - **Triton**: Block-level loops with grid syntax
  - **Pallas**: Sequential loops with scalar tile sizes
  - **CuTe**: Thread-level loops with layout expressions

### 5. **Two-Pass Compilation for Autotuning**
1. **First pass**: Run without optimization (type checking, shape inference)
2. **For each config**: Re-run code generation with specific tile sizes/strategies
3. **Select best**: Benchmark and cache winning config

## Core Data Structures

### HostFunction
Mutable compilation state for a kernel containing:
- `definition`: Source AST, parameters, bound arguments
- `compiler_state`: Symbol provenance, tensor origins, needed imports
- `device_ir`: FX graphs after lowering

### DeviceIR
Backend-agnostic device code representation:
- `graphs`: List of `GraphInfo` objects (root, loops, conditionals)
- Each graph is a FX graph representing a device computation
- Metadata: memory ops, reduction patterns, control flow

### Backend Abstract Class
Minimal interface requiring subclasses to implement:
- **`name`**: Backend identifier (e.g., "triton")
- **`dtype_str(dtype)`**: Map torch.dtype → backend type string
- **`acc_type(dtype)`**: Accumulator type for reductions
- **`function_decorator`**: Decorator for kernel function
- **`constexpr_type`**: Compile-time constant annotation
- **`library_imports`**: Dict of required imports
- Plus optional methods: `customize_ast()`, `pre_codegen()`, config tuning, etc.

## Backend Registration & Codegen Dispatch

### How Backends Are Discovered
```python
# helion/_compiler/backend_registry.py
_BUILTIN_BACKENDS = [TritonBackend, PallasBackend, CuteBackend, TileIRBackend, MetalBackend]

# Each backend registered at module load:
register_compiler_backend(TritonBackend)  # name="triton"
register_compiler_backend(PallasBackend)  # name="pallas"
...

# Runtime lookup:
backend_cls = get_backend_class("triton")
backend = backend_cls()
```

### How Codegen Modules Are Loaded
```python
# Each backend lists its codegen modules:
# helion/_compiler/triton/_codegen_modules.py
from . import memory_ops  # noqa: F401
from . import reduce_ops  # noqa: F401
from . import matmul_ops  # noqa: F401
...

# Each module registers codegen handlers:
# helion/_compiler/triton/memory_ops.py
@_decorators.codegen(store, "triton")  # Register for store() op in Triton
def _(state: CodegenState) -> ast.AST:
    return strategy.codegen_store(state, tensor, subscript, value, mask, cache_modifier)
```

Dispatch flow:
1. `backend_registry.import_backend_codegen()` imports all backend codegen modules
2. Each `@_decorators.codegen(op, "<backend>")` registers the handler on `op`
3. At codegen time: `op.codegen_for_backend(state, backend_name)` → AST

## Extension Point: Adding a New Backend

To add backend "mybackend":

### 1. Create Backend Class
```python
# helion/_compiler/mybackend/backend.py
from helion._compiler.backend import Backend

class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "mybackend"

    def dtype_str(self, dtype: torch.dtype) -> str:
        # Map torch.float32 → "my_f32", etc.
        return {...}[dtype]

    def acc_type(self, dtype: torch.dtype) -> str:
        # Accumulator for reductions (may upcast fp16→fp32)
        return {...}[dtype]

    @property
    def function_decorator(self) -> str:
        return "@my_backend.jit"

    @property
    def constexpr_type(self) -> str:
        return "my_constexpr"

    @property
    def default_launcher_name(self) -> str:
        return "my_kernel_launch"

    @property
    def library_imports(self) -> dict[str, str]:
        return {"my_backend": "import my_backend as my_backend"}
```

### 2. Implement Codegen Modules
```python
# helion/_compiler/mybackend/memory_ops.py
@_decorators.codegen(store, "mybackend")
def _(state: CodegenState) -> ast.AST:
    # Generate backend-specific store statement
    ...

# helion/_compiler/mybackend/reduce_ops.py
@_decorators.codegen(reduce, "mybackend")
def _(state: CodegenState) -> ast.AST:
    # Generate backend-specific reduction
    ...

# Similar for matmul_ops.py, atomic_ops.py, etc.
```

### 3. Register Codegen Modules
```python
# helion/_compiler/mybackend/_codegen_modules.py
from . import memory_ops  # noqa: F401
from . import reduce_ops  # noqa: F401
from . import matmul_ops  # noqa: F401
...
```

### 4. (Optional) ATen Lowering
```python
# helion/_compiler/mybackend/aten_lowering.py
# Map torch.ops.aten.* operations to backend IR via aten_lowering_dispatch
```

### 5. Register Backend
```python
# helion/_compiler/backend_registry.py
_BUILTIN_BACKENDS = [
    TritonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
    MyBackend,  # Add here
]
```

## Compilation Flow by Example: Simple Matmul

```python
@helion.kernel()
def matmul(x, y):
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):  # ← Device loop
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):  # ← Device loop
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
        out[tile_m, tile_n] = acc
    return out
```

### Stages:
1. **Parse**: Extract two `for` loops, `@=` reduction operation
2. **Type Propagate**: tile_m, tile_n → TileIndexType; acc → TensorType[f32]
3. **Device IR Lower**: Extract inner loop into DeviceIR graph (FX)
4. **Inductor Lower**:
   - `x[tile_m, tile_k]` → aten.index.Tensor lowering
   - `@` (matmul) → aten.mm.out lowering
   - `+=` → aten.add.Tensor lowering
5. **Triton Codegen** (example):
   ```python
   @triton.jit
   def matmul_kernel(x_ptr, y_ptr, out_ptr, m, k, n, BLOCK_M, BLOCK_N, BLOCK_K):
       pid_m = tl.program_id(0)
       pid_n = tl.program_id(1)
       tile_m = tl.arange(0, BLOCK_M)
       tile_n = tl.arange(0, BLOCK_N)
       acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
       for tile_k_start in range(0, k, BLOCK_K):
           x_tile = tl.load(x_ptr + ...)
           y_tile = tl.load(y_ptr + ...)
           acc += tl.dot(x_tile, y_tile)
       tl.store(out_ptr + ..., acc)
   ```

## Testing & Debugging

### Run Examples
```bash
cd helion
HELION_AUTOTUNE_EFFORT=none python examples/matmul.py
```

### View Generated Code
```python
import helion
# Prints generated code to stderr/logs
helion.settings.HELION_PRINT_OUTPUT_CODE=1
```

### Enable Debug Logging
```bash
HELION_LOGS=all python your_kernel.py
```

### Test Suite
```bash
pytest test/test_examples.py -k attention -x -vv -s
```

## Summary: Why This Architecture?

1. **Modularity**: Each backend is isolated; adding new backend doesn't touch core compiler
2. **Reusability**: Device IR is shared across backends; only codegen differs
3. **Extensibility**: Lowering objects allow fine-grained op customization per backend
4. **Maintainability**: FX graphs provide standard representation; leverage PyTorch ecosystem
5. **Performance**: Two-pass compilation enables per-config specialization and autotuning
6. **Traceability**: Source locations preserved through all passes for error reporting

---

**Next Step for New Backend**: Study [helion-compilation-flow.md](/memories/repo/helion-compilation-flow.md) in repo memory for detailed API documentation and extension points.
