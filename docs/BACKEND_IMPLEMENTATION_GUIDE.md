# Helion Backend Implementation Quick Reference

## File Structure for New Backend "mybackend"

```
helion/_compiler/mybackend/
├── __init__.py                    # Empty or re-exports
├── backend.py                     # Backend class (required)
├── _codegen_modules.py            # Module registry (required)
├── aten_lowering.py               # ATen op mappings (optional)
├── memory_ops.py                  # Store/load codegen
├── reduce_ops.py                  # Reduction codegen
├── matmul_ops.py                  # Matmul codegen
├── atomic_ops.py                  # Atomic operation codegen
├── debug_ops.py                   # Debug/print codegen
├── creation_ops.py                # Tensor factory codegen
├── view_ops.py                    # View/reshape codegen
├── scan_ops.py                    # Scan/sequential ops
├── barrier.py                     # Barrier/sync codegen
└── printer.py                     # (Optional) Pretty-print expressions
```

## Core Backend Class Template

**File**: `helion/_compiler/mybackend/backend.py`

```python
from helion._compiler.backend import Backend
from helion import exc
import torch

class MyBackend(Backend):
    @property
    def name(self) -> str:
        """Unique backend name for registry."""
        return "mybackend"

    @property
    def experimental(self) -> bool:
        """Whether to warn about experimental status."""
        return False

    def validate_environment(self) -> None:
        """Raise helion.exc.* if backend unavailable."""
        # Check for required libraries, CUDA versions, etc.
        pass

    # ============ TYPE MAPPING (Required) ============

    def dtype_str(self, dtype: torch.dtype) -> str:
        """Map torch.dtype to backend type string."""
        dtype_map = {
            torch.float32: "my_f32",
            torch.float16: "my_f16",
            torch.int32: "my_i32",
            torch.int64: "my_i64",
            # Add all supported dtypes
        }
        if dtype not in dtype_map:
            raise exc.UnsupportedDtype(self.name, dtype)
        return dtype_map[dtype]

    def acc_type(self, dtype: torch.dtype) -> str:
        """Accumulator type for reductions (may differ from dtype)."""
        # Common pattern: upcast fp16 to fp32 for numerical stability
        acc_map = {
            torch.float32: "my_f32",
            torch.float16: "my_f32",  # Upcast!
            torch.int32: "my_i32",
            torch.int64: "my_i64",
        }
        return acc_map.get(dtype, self.dtype_str(dtype))

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        """Index type (offsets, counts). Override if different from dtype_str."""
        return self.dtype_str(index_dtype)

    # ============ CODE GENERATION SETUP (Required) ============

    @property
    def function_decorator(self) -> str:
        """Decorator applied to kernel function."""
        return "@my_backend.jit"

    @property
    def constexpr_type(self) -> str:
        """Annotation for compile-time constants."""
        return "my_backend.constexpr"

    @property
    def default_launcher_name(self) -> str:
        """Default name for host-side kernel launcher."""
        return "my_backend_launch"

    @property
    def library_imports(self) -> dict[str, str]:
        """Map short names to import statements."""
        return {
            "my_backend": "import my_backend",
            "torch": "import torch",
        }

    # ============ OPTIONAL CUSTOMIZATIONS ============

    def customize_ast(self, hf: "HostFunction") -> None:
        """Backend-specific AST rewrites before type propagation."""
        # Example: Rewrite patterns for better code gen on this backend
        # Called after static loop unrolling, before type propagation
        pass

    def pre_codegen(
        self,
        graphs: list["GraphInfo"],
        config: "Config",
        tile_strategy: "TileStrategyDispatch",
    ) -> None:
        """Analyze/transform graphs before code generation."""
        # Called right before GenerateAST, after tiling is finalized
        pass

    def config_value_priors(self, config_spec: "ConfigSpec") -> dict[str, "ValuePrior"]:
        """Bias autotuning search toward good configs."""
        return {}

    def supports_config_key(self, key: str) -> bool:
        """Which tuning knobs does this backend support?"""
        supported = {"num_warps", "block_m", "block_n", "block_k"}
        return key in supported

    # ============ EXPRESSION GENERATION ============

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        """Generate cast expression: my_backend.cast(expr, dtype)"""
        return f"my_backend.cast({expr_str}, {dtype_str})"

    def cdiv_expr(self, numel: str, block_size: str, *, is_device: bool) -> str:
        """Ceiling division: (numel + block_size - 1) // block_size"""
        return f"(({numel}) + ({block_size}) - 1) // ({block_size})"

    def range_str(self, begin: str | None, end: str, step: str | None) -> str | None:
        """Custom range() syntax, or None for Python default."""
        return None

    # ============ DEVICE OPERATIONS (Optional) ============

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        """Get program/block ID for dimension."""
        raise exc.BackendUnsupported(self.name, "program IDs")

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Compute grid index from offset and block size."""
        raise exc.BackendUnsupported(self.name, "grid index")

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        """Compute loop index from offset and block size."""
        raise exc.BackendUnsupported(self.name, "loop index")

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        """Load scalar value from tensor argument."""
        raise exc.BackendUnsupported(self.name, "scalar load")

    def thread_in_tile_mask_expr(self, block_size_var: str, *, axis: int = 0) -> str | None:
        """Optional mask restricting active threads to tile width."""
        return None

    def max_reduction_threads(self) -> int | None:
        """Max threads for warp-level reduction, or None if unlimited."""
        return None

    @staticmethod
    def reserved_launch_param_names() -> frozenset[str]:
        """Names reserved by launcher (can't be used as kernel args)."""
        return frozenset({"grid", "block", "stream"})
```

## Codegen Module Registration

**File**: `helion/_compiler/mybackend/_codegen_modules.py`

```python
"""MyBackend codegen module registry.

Importing this module imports every MyBackend-specific codegen module,
so their @_decorators.codegen(op, "mybackend") handlers register onto
the ops they extend.
"""

from . import aten_lowering  # noqa: F401
from . import atomic_ops  # noqa: F401
from . import creation_ops  # noqa: F401
from . import memory_ops  # noqa: F401
from . import reduce_ops  # noqa: F401
from . import matmul_ops  # noqa: F401
# Add more as needed
```

## Memory Operations Codegen

**File**: `helion/_compiler/mybackend/memory_ops.py`

```python
"""MyBackend codegen for load/store operations."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language.memory_ops import load, store
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(store, "mybackend")
def _(state: CodegenState) -> ast.AST:
    """Generate store operation for MyBackend."""
    tensor = state.proxy_arg(0)          # The tensor being stored to
    subscript = state.proxy_arg(1)       # List of indices
    value = state.ast_arg(2)             # Value to store
    extra_mask = state.ast_args[3]       # Optional mask

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Only tensor stores supported")

    # Get indexing strategy (different per backend)
    indexing_idx = state.device_function.allocate_store_index()
    strategy = state.device_function.get_indexing_strategy(indexing_idx)

    # Use strategy to generate store code
    return strategy.codegen_store(
        state,
        tensor,
        list(subscript),
        value,
        extra_mask,
        cache_modifier=None
    )


@_decorators.codegen(load, "mybackend")
def _(state: CodegenState) -> ast.AST:
    """Generate load operation for MyBackend."""
    tensor = state.proxy_arg(0)          # The tensor being loaded from
    subscript = state.proxy_arg(1)       # List of indices
    extra_mask = state.ast_args[2]       # Optional mask

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Only tensor loads supported")

    indexing_idx = state.device_function.allocate_load_index()
    strategy = state.device_function.get_indexing_strategy(indexing_idx)

    return strategy.codegen_load(
        state,
        tensor,
        list(subscript),
        extra_mask,
        cache_modifier=None
    )
```

## Reduction Operations Codegen

**File**: `helion/_compiler/mybackend/reduce_ops.py`

```python
"""MyBackend codegen for reduction operations."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.reduce_ops import reduce as reduce_op
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(reduce_op, "mybackend")
def _(state: CodegenState) -> ast.AST:
    """Generate reduction operation for MyBackend."""
    input_tensor = state.proxy_arg(0)
    reduction_type = state.const_arg(1)  # "sum", "max", "min", etc.
    output_dtype = state.proxy_arg(2)

    # Example: generate a sum reduction
    if reduction_type == "sum":
        # MyBackend-specific sum reduction syntax
        return expr_from_string(
            "my_backend.sum({input}, dtype={dtype})",
            input=state.ast_arg(0),
            dtype=expr_from_string(state.backend.dtype_str(output_dtype))
        )

    raise NotImplementedError(f"Reduction {reduction_type} not implemented")
```

## Registration in Backend Registry

**File**: `helion/_compiler/backend_registry.py` (update)

```python
# Add to _BUILTIN_BACKENDS:
from .mybackend.backend import MyBackend

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
    MyBackend,  # ← Add here
]
```

## Key CodegenState Methods

```python
class CodegenState:
    # Accessing arguments
    proxy_arg(i: int)          # Get i-th arg as proxy (for analysis)
    ast_arg(i: int)            # Get i-th arg as AST expression (for codegen)
    const_arg(i: int)          # Get i-th arg as constant value

    # Configuration
    config: Config             # Current tuning config
    backend: Backend           # Active backend

    # Device context
    device_function: DeviceFunction  # Kernel state/temporary allocation

    # Code generation
    add_statement(stmt: ast.AST)  # Add statement to output
    tmpvar(prefix="v")         # Generate temporary variable name

    # Tracking
    fx_node: torch.fx.Node     # Source FX node
```

## Important Patterns

### 1. Backend-Specific Tile Layout
Each backend handles tiling differently:
- **Triton**: Block-based grid with program_id
- **Pallas**: Sequential loops with scalar tile sizes
- **CuTe**: Thread-based with layout expressions

Your backend's `TileStrategy` handles this.

### 2. Using Indexing Strategies
```python
# Allocate index slot
idx = state.device_function.allocate_load_index()
strategy = state.device_function.get_indexing_strategy(idx)

# Use strategy to emit load/store
strategy.codegen_load(state, tensor, subscript, mask, cache_modifier)
strategy.codegen_store(state, tensor, subscript, value, mask, cache_modifier)
```

### 3. Handling Masks
Most backends support masking to avoid out-of-bounds access:
```python
# Example: masked load in Triton-like syntax
offset = expr_from_string("{base} + {idx}", base=base_offset, idx=idx_var)
load_expr = expr_from_string(
    "my_backend.masked_load({ptr}, {mask}, {offset})",
    ptr=tensor_ptr,
    mask=extra_mask,
    offset=offset
)
```

### 4. Register Decorators for All Ops
Template for each op codegen module:
```python
from ...language import _decorators
from ...language.some_ops import some_op

@_decorators.codegen(some_op, "mybackend")
def _(state: CodegenState) -> ast.AST:
    # Generate code for some_op on mybackend
    ...
```

## Testing Your Backend

```bash
# Run single example with your backend
HELION_AUTOTUNE_EFFORT=none python examples/matmul.py --backend mybackend

# Run test suite for your backend
pytest test/ -k mybackend -x -vv -s

# View generated code
HELION_PRINT_OUTPUT_CODE=1 python examples/matmul.py --backend mybackend
```

## Common Pitfalls

1. **Missing dtype mapping**: Ensure all torch dtypes used in examples have `dtype_str()` mappings
2. **Incomplete operation codegen**: Some tests require atomic_ops, scan_ops, etc.
3. **Indexing strategy mismatch**: Tile layout must match backend's grid/thread model
4. **Type annotation mismatches**: `constexpr_type` annotation must match backend syntax
5. **Decorator syntax**: Function decorator must be valid Python/backend syntax

## Resources

- Study existing backends: `helion/_compiler/{triton,pallas,cute}/backend.py`
- Example codegen: `helion/_compiler/triton/memory_ops.py`
- Test examples: `helion/examples/matmul.py`, `helion/examples/attention.py`
- Memory reference: `/memories/repo/helion-compilation-flow.md`

---

**Ready to implement?** Start with backend.py + memory_ops.py, test with matmul.py, then expand to other ops.
