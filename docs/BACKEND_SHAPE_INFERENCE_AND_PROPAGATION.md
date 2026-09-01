# Backend Shape Inference and Propagation (MLIR Backend)

This document explains how shapes are inferred, resolved, and propagated in the Helion MLIR backend in this repository.

It focuses on these implementation files:
- `helion_mlir_backend/_compiler/mlir/build_context.py`
- `helion_mlir_backend/_compiler/mlir/codegen.py`
- `helion_mlir_backend/_compiler/mlir/aten_lowering.py`
- `helion_mlir_backend/_compiler/mlir/support/symbolic_shape_restoration.py`

## Why Shape Propagation Matters

The backend emits statically shaped MLIR tensor operations (for example `tensor.extract_slice`, `linalg.matmul`, and `scf.for` iter args).

If shape propagation is inconsistent across phases, failures appear as:
- Torch FakeTensor broadcast/shape errors during ATen helper preprocessing.
- torch-mlir lowering errors (for example `linalg.generic` shape inference mismatch).
- Lighthouse loop verification errors (for example `scf.yield` not equivalent to loop-carried iter arg semantics).

## High-Level Flow

Shape propagation runs through multiple stages:

1. Resolve configured block sizes.
2. Build symbolic mappings between block ids and symbolic shape values.
3. Normalize nested loop body placeholder metadata.
4. Preprocess ATen nodes into helper functions with concrete tensor signatures.
5. Lower loads/stores and loop-carried values using the same resolved shape model.
6. Emit MLIR where loop iter-arg/result shapes must remain equivalent.

## Stage 1: Block Size Resolution

In `MLIRModuleBuilder._resolve_block_sizes` (`codegen.py`), with shared state owned by `BuildContext`:

- `CompileEnvironment.block_sizes` are resolved from config (`from_config`) with `HostFunction` active.
- Concrete sizes are stored in:
  - `_block_id_to_size: dict[int, int]`

Block-id resolution for any index/subscript node (not just this stage) flows
through a single authoritative helper, `resolve_index_descriptor` in
`support/index_meta.py`, which reads Helion's own device-IR metadata
(`meta['tile_with_offset']`, set for every backend) and
`HostFunction.expr_to_origin` (via `CompileEnvironment.get_block_id`) instead
of name- or identity-based heuristic maps. There is no separate
`block_hint_to_id`/`block_symint_to_id` bookkeeping to keep in sync.

## Stage 2: Nested Body Placeholder Repair

In `support.symbolic_shape_restoration.restore_symbolic_shapes_in_bodies`:

- For `_for_loop` body graphs, placeholder `meta["val"]` can become ambiguous after tracing.
- The backend reconstructs concrete placeholder shapes from loop/block shape nodes.
- The same concrete tensor metadata is propagated to aliases and shape-query nodes (for example `_new_var`, `sym_size.int`, and load-dependent metadata).

This prevents nested loop bodies from collapsing to ambiguous or stale dimensions.

## Stage 3: ATen Helper Construction

In `preprocess_aten_nodes` / `_build_aten_subgraph` (`aten_lowering.py`):

- Each ATen node is isolated into a minimal FX subgraph.
- Tensor placeholders are created from concrete fake tensors derived from node metadata.
- Tensor output metadata is inferred by:
  1. Evaluating the op on concrete fake inputs when possible.
  2. Falling back to resolved metadata shape when evaluation is not possible.

This stage is critical because helper function argument/result types become fixed MLIR function signatures.

One normalization step in `_build_aten_subgraph` rewrites stale/mismatched
operand shapes to a common bounded shape before building the helper. This
normalization is restricted to operands that already share the same rank: an
earlier version also rank-expanded lower-rank operands (e.g. a 1-D bias vs a
2-D accumulator) to match, which produced a helper signature the real
call-site MLIR value never actually matched (the value was never expanded) --
breaking every rank-mismatched broadcast add. Genuine rank-broadcast is left
to torch-mlir's own linalg lowering instead. See
`tests/test_reduce_ops.py::test_rank_mismatch_broadcast_add`.

## Stage 4: Concrete Shape Resolution Rules

In `_resolve_dims` (`aten_lowering.py`) and `resolve_index_descriptor`
(`support/index_meta.py`), dimension resolution order is:

1. Literal int index -> scalar (no block id).
2. `meta['tile_with_offset']` (set by Helion's own device-IR pass) -> block id + offset.
3. `CompileEnvironment.get_block_id(symint)` via `HostFunction.expr_to_origin`.
4. `_get_symnode('block_size_N')` key lookup.
5. A load's own tensor-dimension symbol origin (`sym_size.int(tensor, dim)`).
6. Upper-bound clamping against configured loop bounds.
7. Final fallback to `int(dim)`.

This layered strategy makes propagation robust across traced graph rewrites,
and replaces an earlier generation of id()-keyed/name-heuristic maps that were
removed once this single authoritative path covered every historically-buggy
indexing pattern (combined 2D tile, nested reduction, transpose, reordered
store -- see `tests/test_index_descriptor.py`).

## Stage 5: Load/Store Lowering Consistency

In `lowering/load_slice_ops.py` and `lowering/tile_index_ops.py`:

- Slice result sizes are derived using the same shape resolution path used for ATen helpers.
- IV offsets are inferred from symbolic index nodes and block-id mappings.
- A guarded fallback can infer offset IVs by matching active loop block sizes, while avoiding duplicate block-id use per dimension.

The objective is that extracted slice types match helper function expectations exactly.

## Stage 6: Loop-Carried Equivalence Constraints

Nested reduction loops use `scf.for` iter args for accumulators.

Even with correct shape propagation, reduction updates must preserve loop-carried equivalence semantics expected by downstream passes.

In this backend, two custom lowerings are used to preserve this property for common matmul accumulation patterns:

- `aten.addmm` -> `linalg.matmul(... outs=[acc])`
- `acc + torch.matmul(lhs, rhs)` pattern -> `linalg.matmul(... outs=[acc])`

Without these rewrites, generic decomposition patterns can still fail verification in downstream loop passes.

## Common Failure Modes

1. FakeTensor broadcast mismatch in ATen preprocessing.
- Typical cause: stale or ambiguous nested-loop metadata when constructing helper subgraphs.

2. `linalg.generic` shape mismatch during torch-mlir pipeline.
- Typical cause: disagreement between helper signatures and emitted load slice shapes.

3. `scf.yield` / iter-arg equivalence failures in nested reductions.
- Typical cause: decomposition path does not preserve loop-carried accumulator semantics.

## Practical Debugging Checklist

1. Verify config-driven block sizes were resolved as expected.
2. Inspect `BuildContext.block_id_to_size` / `block_id_to_upper_bound` and, for
   any individual index node, call `resolve_index_descriptor` directly.
3. Confirm placeholder metadata repair happened for nested `_for_loop` bodies.
4. Check ATen helper signatures and concrete fake input/output shapes.
5. Compare `tensor.extract_slice` result shapes against helper input types.
6. For reduction patterns, verify loop-carried accumulator update lowering.

## Notes on Scope

This document describes current behavior in this repository's MLIR backend and does not claim parity with all upstream torch-mlir or lighthouse lowering configurations.
