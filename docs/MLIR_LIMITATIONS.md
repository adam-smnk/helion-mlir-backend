# MLIR Backend Limitations and Known Constraints

This document lists current limitations for the MLIR backend in this repository.

## Current State (What Is True Today)

- The backend is experimental.
- CPU execution is supported and validated in tests.
- The MLIR validation suite spans `tests/test_mlir_backend.py`,
  `tests/test_mlir_execution.py`, `tests/test_mlir_integration.py`,
  `tests/test_index_descriptor.py`, `tests/test_reduce_ops.py`, and
  `tests/test_property_kernels.py` (property-based fuzz coverage).
- The current suite contains 160 passing tests.
- Both explicit generate-and-execute flow and direct backend="mlir" flow are exercised.
- All example scripts under `examples/` are kept runnable and are re-verified
  after backend changes (`uv run python examples/<name>.py`).

This replaces earlier "IR-only" descriptions.

## 1) Static-Shape Requirement

Current requirement:
- Kernels should be authored with `static_shapes=True`.
- Inputs should have concrete compile-time shapes.

Why:
- The backend resolves tile/block dimensions to concrete sizes for slice types and loop bounds.

Consequence:
- Highly dynamic shape programs can fail shape propagation/lowering.

## 2) CPU-Only Runtime Path

Current behavior:
- `execute_mlir` runtime path is CPU-oriented.
- CUDA tensors are not supported in this path and are expected to raise errors.

Consequence:
- Use CPU tensors for MLIR backend execution tests and workflows.

## 3) Structured Kernel Form Required

Current requirement:
- Tensor work must be inside Helion device loops (`hl.tile(...)`).

Consequence:
- Arbitrary host-style tensor operations outside device loops are rejected by design.

## 4) Operation Coverage Is Pattern-Dependent

Current reality:
- Operation support is not a fixed whitelist/blacklist in docs.
- Practical support depends on combinations that survive:
  - Helion tracing,
  - backend shape propagation,
  - ATen helper lowering via torch-mlir,
  - downstream lighthouse validation.

Examples validated in current tests include:
- Elementwise add/mul style kernels.
- Fused scale-add style patterns.
- Nested tiled matmul accumulation patterns.

## 5) Nested Reduction Semantics Are Sensitive

Known constraint:
- In nested `scf.forall` + `scf.for` reductions, loop-carried accumulator semantics are strict.

Current backend behavior:
- Custom lowerings are used for common accumulation forms to preserve iter-arg/yield equivalence:
  - `aten.addmm` accumulation form.
  - `acc + matmul(...)` accumulation form.

Consequence:
- Equivalent high-level math can succeed or fail depending on the lowered intermediate form.

## 6) Shape Propagation Can Still Be Fragile for New Patterns

Known risk:
- Some new operator compositions can expose metadata inconsistencies in helper preprocessing.

Mitigation already in place:
- Additional fake-tensor evaluation and symbolic mapping logic improves nested-loop shape propagation.

Consequence:
- New complex patterns may still require targeted fixes.

## 7) Source Availability Requirement

Current requirement:
- Kernel source must be discoverable (`inspect.getsource()` path).

Consequence:
- REPL-defined or dynamically generated kernels can fail with source retrieval errors.

## 8) Diagnostics Quality

Current limitation:
- Some failures surface from torch-mlir/lighthouse with low-level pass diagnostics.
- Messages are improving but may still be non-obvious without IR inspection.

Recommended workflow:
- Use `HELION_MLIR_DUMP_PRE_LOWERING=1`.
- Reproduce with targeted tests in `tests/test_mlir_execution.py`.

## 9) Ragged Combined-Tile Boundary Is Not Dynamically Clamped

Current requirement:
- For a combined multi-dimensional tile (e.g. `for tm, tn in hl.tile([m, n])`),
  every dimension whose block size needs more than one iteration must evenly
  divide that dimension's extent.

Why:
- The outer `scf.forall` emits one statically-sized extract/insert per
  iteration with no per-iteration dynamic clamp for a ragged (partial) last
  tile. A single-dimension `hl.tile()` does not have this restriction; its own
  `tile.end`-based dynamic clamp already handles raggedness correctly.

Consequence:
- Compiling such a kernel raises a clear `UnsupportedOperationError` ("ragged
  combined-tile block size") instead of miscompiling.
- Workaround: choose a block size that evenly divides the dimension, or
  restructure the loop so that dimension needs only one iteration.

## 10) Multi-Output Kernels Are Not Supported

Current requirement:
- A kernel must return a single tensor.

Consequence:
- `return out1, out2` raises a clear `UnsupportedOperationError`
  ("multi-output kernel") at compile time.
- Workaround: split the kernel into separate single-output kernels.

## 11) Backend Architecture Boundary

The MLIR backend is intentionally decoupled from `TritonBackend`. It inherits
from Helion's backend-neutral `Backend` and bypasses Helion's Python AST codegen
and Triton cache-management path. The implementation is split into:

- `lowering/` for operation and control-flow emission.
- `aten_bridge/` for custom ATen handling and torch-mlir helper management.
- `support/` for shape repair, type conversion, dispatch, and diagnostics.

This means Triton-specific code-generation behavior is not a fallback for MLIR;
unsupported MLIR operations must be added to the appropriate MLIR lowering or
ATen bridge module.

## Out of Scope for This Backend Today

- Full dynamic-shape-first lowering model.
- GPU runtime execution path parity with CPU path in this backend.
- Guaranteed support for all ATen programs independent of pattern shape.

## Related Docs

- `docs/MLIR_USAGE.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `docs/MLIR_DESIGN.md`
