# MLIR Backend Limitations and Known Constraints

This document lists current limitations for the MLIR backend in this repository.

## Current State (What Is True Today)

- The backend is experimental.
- CPU execution is supported and validated in tests.
- The main validation suite is `tests/test_mlir_execution.py`.
- Both explicit generate-and-execute flow and direct backend="mlir" flow are exercised.

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

## Out of Scope for This Backend Today

- Full dynamic-shape-first lowering model.
- GPU runtime execution path parity with CPU path in this backend.
- Guaranteed support for all ATen programs independent of pattern shape.

## Related Docs

- `docs/MLIR_USAGE.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `docs/MLIR_DESIGN.md`
