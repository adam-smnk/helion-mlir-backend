# MLIR Backend Limitations and Known Constraints

This document lists current limitations for the MLIR backend in this repository.

## Current State (What Is True Today)

- The backend is experimental.
- CPU execution is supported and validated in tests.
- The MLIR validation suite spans `tests/test_mlir_backend.py`,
  `tests/test_mlir_execution.py`, `tests/test_mlir_integration.py`,
  `tests/test_index_descriptor.py`, `tests/test_reduce_ops.py`, and
  `tests/test_property_kernels.py` (property-based fuzz coverage).
- The current suite contains 191 passing tests.
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
- Two-operand `torch.einsum` contractions (see below).

### `torch.einsum`

A two-operand einsum is captured before PyTorch's dispatcher decomposes it and
emitted as one `linalg.contract`, but only when the equation matches that op's
semantics: two operands, no ellipsis, no repeated subscript within an operand
or in the output, every output subscript present in some input, and every
contracted subscript present in both inputs. Multiple contracted dimensions
are supported.

Broadcasting by *omitting* a subscript from one operand (`"bmk,kn->bmn"`) is
supported and costs nothing — the dimension is simply absent from that
operand's indexing map. Broadcasting a *size-1* dimension against a larger one
is not: `linalg.contract` requires matching extents, so such equations stay on
the decomposed path. Ellipsis (`"...ij,...jk->...ik"`) is rejected outright.

Shared subscripts are compared by *symbol* rather than by value, so tile
extents are matched without evaluating them (which would install a shape guard
on an unbacked SymInt). When equality cannot be proven and one side is
statically 1, the equation is conservatively left to PyTorch, since that is the
case einsum would silently broadcast.

Anything else falls back to PyTorch's decomposition, which is correct but
subject to the coverage of the resulting primitives — e.g. `"kk,kn->kn"`
decomposes to `aten.diagonal`, which Helion's shared lowering pass rejects.
Reduction-free equations (`"ij,ij->ij"`, `"m,n->mn"`) are excluded from the
direct path on purpose so they keep their elementwise lowering.

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

## 10) Multi-Output Kernels

Current behavior:
- Every host tensor a phase stores into is an output of that phase's function,
  in first-store order. Each is threaded as its own SSA value, so outputs need
  not share shapes, and independent top-level loops with different grids in one
  phase are separate `scf.forall`s emitted in sequence.

Consequence:
- `execute_mlir` returns a `list` of tensors (not a single tensor) whenever
  there is more than one output.
- Which tensors are returned is decided by what the kernel writes, not by its
  `return` statement (backend review plan, Phase 4).

## 11) Multi-Phase Kernels (`hl.barrier()`) and Host-Tensor Interop

Current behavior:
- Kernels using `hl.barrier()` between top-level `hl.tile()`/`hl.grid()` loops,
  and kernels that read a host-computed tensor beyond their own declared
  parameters (e.g. `scale = x.mean() * 100.0` then `hl.load(scale, [])`), are
  supported -- but **only** through the direct `@helion.kernel(backend="mlir")`
  call path, not through `generate_mlir()`/`execute_mlir()`.

Why:
- Each `hl.barrier()`-separated phase compiles to its own MLIR module/JIT'd
  function. A real host-side "driver" (built from the kernel's own AST, with
  device loops and barriers neutralized) runs between phase calls, threading
  real tensors by host variable name -- this driver only exists on the direct
  call path.

Consequence:
- `generate_mlir()`/`execute_mlir()` raise a clear `UnsupportedOperationError`
  ("multi-phase or host-tensor-interop kernel") for such kernels, naming the
  direct call path as the alternative.
- A phase's output tensor must resolve to a plain host variable name (a
  computed/expression origin is not yet supported).
- The kernel's own final `return` must be a plain name or a tuple/list of
  names (not an arbitrary expression).
- Each phase is compiled as its own fully separate MLIR module (simpler and
  safer than one shared multi-entry-point module); this is a compile-time-only
  cost, cached per kernel/config like any other compile.
- No statement other than `hl.barrier()` itself may appear between two
  top-level device loops (a Helion frontend rule, not backend-specific), so a
  host tensor a later phase needs must be computed before the loop that
  first uses it, not between phases.

Example:
- `examples/multi_phase_mlir.py` -- a runnable two-phase kernel combining
  `hl.barrier()` with a host-computed interop tensor.

## 12) Backend Architecture Boundary

The MLIR backend is intentionally decoupled from `TritonBackend`. It inherits
from Helion's backend-neutral `Backend` and bypasses Helion's Python AST codegen
and Triton cache-management path. The implementation is split into:


- `lowering/` for operation and control-flow emission.
- `aten_bridge/` for custom ATen handling and torch-mlir helper management.
- `support/` for shape repair, type conversion, dispatch, and diagnostics.

This means Triton-specific code-generation behavior is not a fallback for MLIR;
unsupported MLIR operations must be added to the appropriate MLIR lowering or
ATen bridge module.

## 13) Implicit Transpose in a Store Is Rejected

A store must index its destination in the same dimension order the value was
loaded. Helion traces an implicit transpose such as

```python
for tile_p, tile_k, tile_n in hl.tile([panels, k, bn]):
  out[tile_p, tile_k, tile_n] = source[tile_k, tile_p, tile_n]  # rejected
```

without reporting an error, but the traced value has shape
`[tile_k, tile_p, tile_n]` while the destination slice needs
`[tile_p, tile_k, tile_n]`. The backend raises
`UnsupportedOperationError("store with transposed or mismatched tile layout")`
rather than emitting an out-of-bounds `tensor.parallel_insert_slice`.

Reorder explicitly instead:

```python
out[tile_p, tile_k, tile_n] = source[tile_k, tile_p, tile_n].permute(1, 0, 2)
```

The equivalent `hl.grid()`/`hl.tile()` spellings that index the destination in
load order (for example `grid(panel) + tile([k, n])`) are supported directly.

Note that `hl.grid()` loops are not tunable and consume no `block_sizes` slot;
only `hl.tile()` loops do. Supplying a config sized for the grid loops silently
assigns the wrong block size to the tiled loops.

## 14) Optimizing Pipeline Requires Tiles of at Least 32

Under `HELION_MLIR_PIPELINE=1`, lighthouse's cache-level tile-and-fuse assigns a zero tile
size to every dimension smaller than its 32-element cache tile. An op whose tiled dimensions
are all smaller than 32 then aborts the process inside upstream MLIR
(`applyTilingToAll`: "Mismatched number of loops"). This is independent of the backend:
`scripts/lighthouse_small_tile_repro.py N` reproduces it with a single
`linalg.elementwise` on `tensor<Nxf32>` (N < 32 aborts, N >= 32 completes).

Workaround until lighthouse is fixed: use block sizes of at least 32 for kernels lowered
through the optimizing pipeline. The scalar pipeline is unaffected.

## 15) Lighthouse Pipeline Deviations

The local lighthouse checkout carries these pipeline changes (to be upstreamed):
- `bufferization.yaml` runs one-shot bufferization with `allow-return-allocs-from-loops`, so
  loop-carried values no longer have to be updated in place (previously any such update
  failed with "Yield operand #0 is not equivalent to the corresponding iter bbArg").
- `scalar-lowering.yaml` includes `bufferization-cleanup.yaml` to deallocate the buffers that
  option allows inside loops.

## Out of Scope for This Backend Today

- Full dynamic-shape-first lowering model.
- GPU runtime execution path parity with CPU path in this backend.
- Guaranteed support for all ATen programs independent of pattern shape.

## Related Docs

- `docs/MLIR_USAGE.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `docs/MLIR_DESIGN.md`
