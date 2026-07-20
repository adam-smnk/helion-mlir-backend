# MLIR Backend - Limitations and Known Issues

## Current Status

**Version:** MVP (Minimum Viable Product)

The MLIR backend is a proof-of-concept implementation that successfully generates valid MLIR IR from Helion kernels. It is suitable for experimentation and IR generation, but not yet feature-complete for all Helion use cases.

## Static Shapes Only

### Limitation
- **Only static shapes are supported**
- Dynamic tensor dimensions (SymInt) are not fully supported in the lowering
- All tensor shapes must be known at compile time

### Why
- MLIR's tensor abstraction can represent dynamic dimensions (`?`)
- However, the current lowering assumes concrete sizes for index arithmetic
- Proper SymInt support requires symbolic execution in MLIR IR

### Workaround
Always provide concrete tensor shapes:

```python
# GOOD - concrete shapes known
x = torch.randn(64, 128, dtype=torch.float32)
y = torch.randn(128, 96, dtype=torch.float32)
module = generate_mlir(kernel, [x, y])

# BAD - dynamic shapes not supported
x = torch.randn(n, k, dtype=torch.float32)  # n, k are variables
module = generate_mlir(kernel, [x, y])  # May fail
```

### Future Work
- Support symbolic dimensions in index arithmetic
- Generate MLIR with dynamic extent markers
- Implement SymInt constant folding

## Limited Operation Support

### Fully Supported

| Category | Operations |
|----------|------------|
| Linear Algebra | matmul, addmm |
| Element-wise | add, mul, relu, abs |
| Indexing | extract_slice, insert_slice, getitem |
| Utilities | zeros, full, constant creation |

### Partially Supported

| Operation | Status | Notes |
|-----------|--------|-------|
| max/min | linalg.generic | Element-wise only, no reduction |
| exp/log | linalg.generic | Element-wise only |
| transpose | Not tested | May work but not validated |
| reshape | Not supported | Would require bufferization |

### Not Supported

| Operation | Reason | Workaround |
|-----------|--------|-----------|
| Layer Normalization | Not implemented | Decompose into element-wise ops |
| Softmax | Not implemented | Implement as exp + sum |
| Attention | Not implemented | Use matmul + softmax sequence |
| Group Operations | Not implemented | Reshape + elementwise |
| Reductions | Not implemented | Use accumulator loops |
| Scatter/Gather | Not implemented | Not applicable to tensor IR |
| Dynamic reshaping | Fundamentally incompatible | Use fixed shapes |
| Sparse operations | Not applicable | Dense tensors only |

### Implementation Status

**Implemented (codegen.py lines 395+):**
- `_lower_host_tensor` - Parameter lookup
- `_lower_get_symnode` - Block ID constants
- `_lower_sym_size_int` - Tensor dimension extraction
- `_lower_full` - Constant tensor creation
- `_lower_zeros` - Zero tensor creation
- `_lower_for_loop` - Sequential scf.for loops
- `_lower_load` - tensor.extract_slice
- `_lower_store` - Accumulate slices for insertion
- `_lower_addmm` - Matrix multiply accumulation
- `_lower_mm` - Matrix multiply
- `_lower_relu` - ReLU activation
- `_lower_add` - Element-wise addition
- `_lower_mul` - Element-wise multiplication
- `_lower_getitem` - Extract loop result
- `_lower_phi` - SSA value passing
- `_lower_new_var` - Variable renaming

**Not Implemented:**
- Softmax
- Layer normalization
- Attention patterns
- Exponential/logarithm (linalg.generic support exists but not wired)
- Reductions (sum, mean, max pooling)
- Transpose/permutation
- Convolution
- Any sparse operations

## Memory and Bufferization Constraints

### Limitation
- No explicit memory management
- No bufferization phase in backend
- All operations on abstract tensors

### Impact
- Generated MLIR cannot be directly executed
- Requires downstream compiler passes (Triton, MLIR-to-LLVM)
- Memory layout optimization deferred to downstream

### Design Decision
- Keep backend focused on high-level IR generation
- Let downstream compilers handle bufferization and memory optimizations
- Maintains separation of concerns

### Workaround
- Use downstream compiler (Triton recommended) for compilation to executable code
- Or implement your own bufferization pass if needed

## Device Constraints

### Limitation
- Target device is CPU by default
- CUDA device support not tested
- No explicit device-specific operations

### Current Behavior
```python
device = torch.device("cpu")  # Supported
x = torch.randn(64, 64, device=device)
module = generate_mlir(kernel, [x])  # Works

device = torch.device("cuda:0")  # May not work properly
x = torch.randn(64, 64, device=device)
module = generate_mlir(kernel, [x])  # Untested
```

### Future Work
- Test and validate CUDA/GPU device handling
- Add device-specific annotations to MLIR IR
- Support hardware-specific optimization hints

## Scalar Argument Limitations

### Limitation
- Limited support for non-tensor scalar arguments
- Float/int literals must be converted to tensors or embedded as constants

### Current Behavior
```python
# Works: Tensor arguments
@hl.kernel
def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

# Partial: Boolean/constant scalars
@hl.kernel
def kernel(x: torch.Tensor) -> torch.Tensor:
    const_scale = 2.0  # Compiled as constant
    return x * const_scale

# Doesn't work well: Dynamic scalar arguments
@hl.kernel
def kernel(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale  # scale not properly traced
```

### Workaround
- Convert scalars to single-element tensors
- Embed constants directly in kernel
- Use hl.zeros() for dynamic size initialization only

## Error Message Quality

### Limitation
- Limited error diagnostics
- Compilation failures may have cryptic messages
- No suggestions for fixing common issues

### Examples
```
ValueError: invalid literal for int() with base 10: 'zuf0'
# Not helpful - should say "unsupported operation" or similar

NoDeviceLoopsInKernel: (message about tile loops)
# Good - clearly indicates the problem

AssertionError at codegen.py:439
# Not helpful - should identify the problematic operation
```

### Workaround
- Check [MLIR_USAGE.md](MLIR_USAGE.md) troubleshooting section
- Refer to kernel patterns in [examples/](../examples/)
- Examine test cases in [tests/test_mlir_backend.py](../tests/test_mlir_backend.py)

## Type System Limitations

### Supported Types
See [Type System](MLIR_USAGE.md#supported-data-types)

### Not Supported
- Complex numbers
- Custom numeric types
- Struct/record types
- References/pointers

### Type Mixing
```python
# Works: Consistent dtypes
x = torch.randn(64, 64, dtype=torch.float32)
y = torch.randn(64, 64, dtype=torch.float32)
z = x + y

# Doesn't work: Mixed dtypes
x = torch.randn(64, 64, dtype=torch.float32)
y = torch.randn(64, 64, dtype=torch.float64)
z = x + y  # Type mismatch error expected
```

## Control Flow Limitations

### Supported
- `hl.tile()` loops over static dimensions
- `hl.grid()` loops over block dimensions
- Sequential `scf.for` loops inside tiles

### Not Supported
- Conditional branches (if/else)
- While loops with dynamic conditions
- Switch statements
- Exceptions/error handling

### Impact
Kernels must be "structured" without dynamic control flow:

```python
# GOOD - structured control flow
for i in hl.tile(n):
    out[i] = compute(x[i])

# BAD - dynamic control flow not supported
if x > threshold:
    out = x + 1
else:
    out = x - 1
```

## Testing and Validation

### Test Coverage
- ✅ Type conversions (5/5 passing)
- ✅ Basic operations (3/3 passing)
- ✅ Full kernel compilation (3/3 passing)
- ✅ MLIR syntax validity (2/2 passing)
- ⚠️ Error handling (2/2 framework ready but needs real errors)

### Gaps
- No end-to-end execution tests (MLIR IR only, not runnable)
- No downstream compiler validation
- No performance benchmarks
- Limited edge case coverage

### Future
- Integration tests with Triton compilation
- Performance profiling
- Extended operation coverage validation

## Known Issues

### Issue 1: Device Loop Requirement
**Status:** By design

Helion requires all tensor operations inside tile loops. This is not a MLIR backend limitation but a Helion language design choice.

```
Error: NoDeviceLoopsInKernel: Tensor operation outside device loops
Fix: Wrap operation in hl.tile() loop
```

### Issue 2: Closure Variables in Kernels
**Status:** Helion limitation

Cannot use captured variables from outer scope:

```python
scale = 2.0
@hl.kernel
def kernel(x: torch.Tensor) -> torch.Tensor:
    return x * scale  # Error: Closure not supported
```

**Fix:** Embed constant or pass as argument

### Issue 3: MLIR Context Requirement
**Status:** Expected, properly handled

MLIR operations require active Context and Location. The backend handles this internally.

### Issue 4: Source Code Requirement
**Status:** Expected

Cannot trace kernels defined interactively (REPL, stdin, dynamic exec).

**Fix:** Define kernel in proper Python file with `inspect.getsource()` access

## Performance Characteristics

### Current
- IR generation: 5-10ms for typical kernels
- IR size: 5-20KB for small kernels (grows with complexity)
- Memory: Minimal (in-process Python objects)

### Bottlenecks
- Not applicable (generation only, not execution)
- Downstream compiler performance depends on specific toolchain

## Roadmap

### Short Term (MVP → Beta)
- [ ] Improve error messages with actionable suggestions
- [ ] Add softmax and layer norm support
- [ ] Validate CUDA device handling
- [ ] Extend test coverage

### Medium Term
- [ ] Dynamic shape support (SymInt)
- [ ] Attention operation support
- [ ] Integration tests with Triton
- [ ] Performance benchmarks

### Long Term
- [ ] Custom operation framework
- [ ] Multiple downstream target support
- [ ] Advanced fusion patterns
- [ ] Profile-guided optimization hints

## Getting Help

1. **Check Examples**: [examples/](../examples/)
2. **Read Usage Guide**: [MLIR_USAGE.md](MLIR_USAGE.md)
3. **Review Tests**: [tests/test_mlir_backend.py](../tests/test_mlir_backend.py)
4. **Inspect Design**: [MLIR_DESIGN.md](MLIR_DESIGN.md)
5. **Report Issues**: Include kernel code, error message, and expected behavior
