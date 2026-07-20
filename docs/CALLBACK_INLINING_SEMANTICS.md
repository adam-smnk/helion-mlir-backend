# Helion Callback Inlining: Bridging Python Dynamics to Concrete Backend Code

## Executive Summary

Helion bridges the semantic gap between dynamic Python kernels with callbacks/closures and concrete backend code (Triton, Pallas, CuTe) through **inline tracing during device IR lowering**. Callbacks are **not** passed as function pointers to generated kernels—they are completely inlined at compile time.

**Key Result**: For each autotuned configuration, callbacks produce a single concrete kernel with no function calls or dynamic dispatch. All epilogue logic becomes backend-specific primitives specialized to that config's block sizes.

---

## Problem: Abstraction Mismatch

### Helion Kernel (Dynamic Semantics)
```python
@helion.kernel()
def matmul(x, y, epilogue: Callable = lambda acc, tile: acc):
    for tile_m, tile_n in hl.tile([m, n]):
        acc = ...  # Compute
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
```

**Features**:
- Callbacks as parameters with dynamic behavior
- Closures capturing host variables (e.g., `bias`)
- Symbolic block sizes during parsing
- Python syntax/semantics (if/for/while, tensor indexing, etc.)

### Triton Kernel (Static Semantics)
```python
@triton.jit
def _helion_matmul(x_ptr, y_ptr, out_ptr, bias_ptr, BLOCK_M, BLOCK_N, BLOCK_K):
    # No function pointers, no dynamic dispatch
    # All block sizes are concrete literals
    # Pure Triton primitives: tl.load, tl.store, tl.where, etc.
```

**Constraints**:
- Statically-typed, no function pointers
- Concrete block sizes (JIT-compiled once per size)
- Limited to backend-specific operations
- No Python runtime

### The Bridge: Inline Tracing

Helion solves this by executing callbacks **during compilation** in a special mode where operations record themselves as FX graph nodes instead of executing for real. The result is a concrete kernel with all callback logic inlined.

---

## Three-Stage Compilation: Callback Path

### Stage 1: Closure Lifting (Parse Time)

**Location**: `helion/_compiler/lift_closures.py`

```python
def lift_closures(func: FunctionType, origin: Origin) -> FunctionType:
    """Wrap closure references so they're captured during tracing."""
```

**What happens**:
```python
# User writes:
bias = torch.randn(n, dtype=torch.float32)
epilogue_lambda = lambda acc, tile: torch.relu(acc + bias[tile[1]])

# Closure lifting wraps function so that when `bias` is accessed:
# 1. HostFunction.current().register_fake(
#        variable_name="bias",
#        value=bias_tensor,
#        origin=ClosureOrigin(epilogue_closure, index=0)
#    )
# 2. Closure is marked read-only (mutations raise error)
# 3. Variable is tracked with origin for later code generation
```

**Result**: Closure variables are extracted and tracked with `ClosureOrigin`, which knows:
- Which closure they came from (`epilogue.__closure__`)
- Their index in the closure tuple
- Original value and deriving symbol origin

### Stage 2: Device IR Tracing (Type Propagation + Lowering)

**Location**: `helion/_compiler/device_ir.py::WalkDeviceAST::visit_Call` (line 2872)

```python
def visit_Call(self, node: ast.Call) -> object:
    args = [self.visit(arg) for arg in node.args]
    func = self.visit(node.func)
    
    # KEY: Direct call, NOT an FX node
    return _CheckForIndexCalls.retry_call(func, args, kwargs)
```

**When callback is called during tracing**:

```python
# User code in kernel loop:
out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))

# During tracing:
# 1. func = the actual lambda (retrieved from CallableType)
# 2. args[0] = acc_proxy (a torch.fx.Proxy wrapping a tensor)
# 3. args[1] = (tile_m_proxy, tile_n_proxy) (tile index proxies)
#
# 4. Lambda EXECUTES in tracing mode:
#    lambda acc, tile: torch.relu(acc + bias[tile[1]])
#
# 5. Each operation records itself as an FX node:
#    - bias[tile[1]]     → FX: load(bias, tile_index)
#    - acc + ...         → FX: add(acc, loaded_bias)
#    - torch.relu(...)   → FX: relu(add_result)
```

**Result**: Device IR FX graph contains inlined operations, NOT a "call_epilogue" node.

**Example FX graph (conceptual)**:
```
placeholder: acc, tile_m, tile_n
placeholder: bias_host_ref
call_function: load(bias_host_ref, index=tile_n)
call_function: add(acc, load_result)
call_function: relu(add_result)
output: relu_result
```

### Stage 3: Backend Codegen (Config Specialization)

**Location**: `helion/_compiler/generate_ast.py`, backend-specific codegen (e.g., `helion/_compiler/triton/`)

For each autotuned config:

```python
config = Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)

# GenerateAST re-runs with concrete block sizes
# Each inlined operation specializes to config values
```

**Generated Triton (for BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)**:
```python
@triton.jit
def _helion_matmul(
    x_ptr, y_ptr, out_ptr,
    x_stride_0, x_stride_1,  y_stride_0, y_stride_1,
    m, k, n,
    bias_ptr,  # ← Lifted from epilogue.__closure__
    # BLOCK_M, BLOCK_N, BLOCK_K are compile-time constants, not parameters
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block iteration...
    for tile_k_start in range(0, k, 32):  # BLOCK_K is literal 32
        # Load, compute
        acc += compute_tile()
    
    # Inlined epilogue (NO function call):
    # Instead of: acc = epilogue(acc, (tile_m_idx, tile_n_idx))
    
    tile_m_idx = pid_m * 128 + tl.arange(0, 128)  # BLOCK_M is literal 128
    tile_n_idx = pid_n * 128 + tl.arange(0, 128)  # BLOCK_N is literal 128
    
    # Load bias[tile_n_idx]
    bias_val = tl.load(bias_ptr + tile_n_idx)
    
    # Compute: relu(acc + bias)
    acc_plus_bias = acc + bias_val
    acc_relu = tl.where(acc_plus_bias > 0, acc_plus_bias, 0.0)
    
    # Store
    out_offset = (pid_m * 128 + tl.arange(0, 128)) * stride_m + \
                 (pid_n * 128 + tl.arange(0, 128)) * stride_n
    tl.store(out_ptr + out_offset, acc_relu)
```

**Observations**:
- No `epilogue` parameter in kernel signature ✓
- No function calls or function pointers ✓
- `bias_ptr` comes from closure lifting ✓
- Block sizes are concrete literals for Triton JIT ✓
- All epilogue operations are inlined Triton primitives ✓

---

## Parameter Transformation: Helion → Triton

| Component | Helion Kernel | Device IR (FX) | Triton Kernel |
|-----------|--------------|-----------------|---------------|
| **Tensor Arguments** | `x: Tensor, y: Tensor` | FX inputs with metadata | `x_ptr, y_ptr, x_stride_*, y_stride_*` |
| **Block Sizes** | `BLOCK_M=hl.symbol()` (symbolic) | Resolved via config | `BLOCK_M` (concrete literal, not param) |
| **Epilogue Parameter** | `epilogue: Callable` (Python function) | Executed inline; no node | **DOES NOT EXIST** |
| **Epilogue Operations** | In callback body | FX nodes (load, add, relu) | Inlined Triton ops (`tl.load`, `tl.add`, `tl.where`) |
| **Closure Variables** | In `epilogue.__closure__` | Lifted with `ClosureOrigin` | `bias_ptr` (kernel parameter) |

---

## Type System: CallableType

**Location**: `helion/_compiler/type_info.py`

```python
class CallableType(LiteralType):
    """Type representation for callable values."""
    value: Callable[..., object]  # The actual function object
    
    def propagate_call(self, args, kwargs, origin):
        """When called in device code: trace execution and infer return type."""
```

**During type propagation**:

```python
# epilogue: CallableType(value=lambda acc, tile: ...)
# Call in kernel: result = epilogue(acc, (tile_m, tile_n))

# Type inference:
# 1. args[0]: TensorType[f32, {BLOCK_M, BLOCK_N}]
# 2. args[1]: SequenceType(TileIndexType, TileIndexType)
# 3. epilogue.propagate_call(args, origin=DeviceOrigin(...))
#    → Execute lambda in tracing mode
#    → Infer return: TensorType[f32, {BLOCK_M, BLOCK_N}]
```

**Key**: Callbacks are deterministic, side-effect-free during tracing. Mutating a closure variable raises an error.

---

## Closure Origin Tracking

**Location**: `helion/_compiler/variable_origin.py`

```python
@dataclasses.dataclass
class ClosureOrigin(WrappedOrigin):
    key: int  # Index in __closure__ tuple
    
    def needs_rename(self) -> bool:
        return True  # Must be renamed in generated code
    
    def host_str(self) -> str:
        # E.g., "epilogue.__closure__[0].cell_contents"
        return f"{self.value.host_str()}.__closure__[{self.key}].cell_contents"
```

**Example**:
```python
def kernel(x, y, epilogue):
    # During tracing, when 'bias' is accessed inside epilogue:
    bias_origin = ClosureOrigin(
        value=ArgumentOrigin("epilogue"),  # The parameter
        key=0                               # First closure cell
    )
    
    # Codegen uses this to emit:
    # bias_ptr = epilogue.__closure__[0].cell_contents._data_ptr()
```

---

## Why This Works: A Concrete Example

### User Writes:
```python
bias = torch.randn(n, dtype=torch.float32)
x_scaled = x * 2.0

@helion.kernel()
def complex_matmul(x, y, epilogue=None):
    if epilogue is None:
        epilogue = lambda acc, tile: acc
    
    for tile_m, tile_n in hl.tile([m, n]):
        acc = torch.zeros(...)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[...], y[...])
        
        # Dynamic epilogue with closures
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))

# Call with complex epilogue capturing multiple variables
result = complex_matmul(
    x_scaled, y,
    epilogue=lambda acc, tile: torch.relu(acc + bias[tile[1]]) * 0.5
)
```

### What Gets Generated:

**Kernel signature** (only kernel args, no epilogue):
```python
def _kernel(
    x_ptr, y_ptr, out_ptr,           # Main inputs
    x_stride_*, y_stride_*,           # Strides
    bias_ptr,                         # Lifted from closure
    scale_factor=0.5,                 # Literal constant
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32  # Config constants (not params)
)
```

**Epilogue code** (inlined):
```python
# Load bias
bias_val = tl.load(bias_ptr + tile_n_idx)

# Compute: relu(acc + bias[tile_n]) * 0.5
temp = acc + bias_val
relu_result = tl.where(temp > 0, temp, 0.0)
final = relu_result * 0.5

# Store
tl.store(out_ptr + out_offset, final)
```

### Result:
- ✅ No epilogue function passed
- ✅ No dynamic dispatch
- ✅ Concrete kernel for each config (128x128x32 version, 64x64x32 version, etc.)
- ✅ All epilogue ops inlined and specialized

---

## Implications for New Backend Implementation

### 1. Don't Expect Callbacks in Kernel Signature
Backends receive only the **lifted closure variables** as kernel arguments, never the callback itself.

### 2. Inlining is Automatic
Device IR lowering already handles inlining via `visit_Call`. Your backend codegen just walks the FX graph.

### 3. Focus on Closure Variable Handling
```python
# Your backend needs to:
# 1. Accept lifted variables: bias_ptr, scale_factor, etc.
# 2. Map them to backend memory (GPU pointers, register values, etc.)
# 3. Use them in inlined operations
```

### 4. No Special Callback Trampolines Required
Unlike some compilation systems (e.g., OpenMP callbacks), Helion doesn't need runtime function dispatch because everything is inlined at compile time.

### 5. Block Size Specialization Applies to Epilogue
```python
config = Config(BLOCK_M=256, BLOCK_N=256, BLOCK_K=64)

# When GenerateAST re-runs with this config:
# - All tl.arange(0, BLOCK_M) becomes tl.arange(0, 256)
# - All index calculations adapt
# - Epilogue operations are re-inlined with concrete sizes
```

---

## Configuration Variability

For the same kernel, different configs produce different epilogue code:

| Config | Generated Tile Indexing | Epilogue Specialization |
|--------|------------------------|-----------------------|
| `BLOCK_M=64, BLOCK_N=64` | `tl.arange(0, 64)` | Inlined ops assume 64² tile |
| `BLOCK_M=128, BLOCK_N=128` | `tl.arange(0, 128)` | Inlined ops assume 128² tile |
| `BLOCK_M=256, BLOCK_N=128` | Mixed ranges | Inlined ops assume 256×128 tile |

The autotuner tries different configs; each produces a separately-JIT-compiled kernel with its own inlined epilogue specialization.

---

## Comparison: Other Backends

This inlining strategy is consistent across all Helion backends:

| Backend | Callback Inlining | Closure Variables | Result |
|---------|-------------------|-------------------|--------|
| **Triton** | Inline via `tl.load`, `tl.where`, etc. | Device pointers (`*_ptr`) | `@triton.jit` with no calls |
| **Pallas** | Inline via `pl.store`, `pl.load`, etc. | HBM references | Concrete Pallas kernel |
| **CuTe** | Inline via CuTe layout algebra | Tensor descriptors | CuTe reference kernel |
| **Metal** | Inline via Metal primitives | Argument buffers | Metal compute shader |

Each backend receives the same FX graph with inlined operations and translates it to native primitives.

---

## Key Files for Understanding Callback Inlining

| File | Role |
|------|------|
| `helion/_compiler/lift_closures.py` | Extract closure variables before tracing |
| `helion/_compiler/variable_origin.py` | Track `ClosureOrigin` for code generation |
| `helion/_compiler/device_ir.py::WalkDeviceAST::visit_Call` | Execute callback inline during tracing |
| `helion/_compiler/type_info.py::CallableType` | Type representation and tracing logic |
| `helion/_compiler/generate_ast.py` | Walk FX graph and call backend codegens |
| `helion/_compiler/triton/` (backend-specific) | Translate inlined FX ops to Triton code |

---

## Verification: Debugging Callback Inlining

To verify callbacks are inlined in your kernel:

1. **Print Device IR**:
   ```python
   env = CompileEnvironment.current()
   print(env.device_ir)  # Look for call_function nodes; epilogue calls should be absent
   ```

2. **Check Generated Triton**:
   ```python
   kernel = compile_kernel(...)
   print(kernel.asm)  # No indirect calls
   ```

3. **Trace Closure Lifting**:
   ```python
   from helion._compiler.lift_closures import lift_closures
   wrapped_fn = lift_closures(user_fn, origin)
   # wrapped_fn will raise if closure is mutated during tracing
   ```

---

## Conclusion

Helion's callback inlining mechanism elegantly solves the abstraction mismatch between dynamic Python kernels and static backend code by:

1. **Capturing closures** at parse time (extraction + origin tracking)
2. **Executing callbacks inline** during device IR lowering (via tracing)
3. **Specializing to each config** at codegen time (concrete block sizes)

The result is a **single concrete kernel per config** with all callback logic inlined, no function pointers, and optimal specialization for the chosen tile sizes. This is the key to Helion's ability to bridge Python's dynamic semantics with static backend compilation while maintaining performance.
