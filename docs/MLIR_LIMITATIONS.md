# MLIR Backend Limitations and Known Constraints

This document lists current limitations for the MLIR backend in this repository.

## Current State (What Is True Today)

- The backend is experimental.
- CPU execution is supported and validated in tests.
- The MLIR validation suite lives under `tests/` (unit, execution, golden-IR,
  AMX-gate and property-based tests; `uv run pytest tests/`).
- Direct `backend="mlir"` calls, `compile_mlir`, and the explicit
  `generate_mlir` + `execute_mlir` flow are exercised.
- All example scripts under `examples/` are kept runnable and are re-verified
  after backend changes (`uv run python examples/<name>.py`).

This replaces earlier "IR-only" descriptions.

## 1) Dynamic Shapes

Current behavior:
- `static_shapes=True` kernels are fully static.
- `static_shapes=False` kernels compile once per config and Helion shape bucket and
  run for any sizes in it. Sizes known at compile time (including `hl.specialize`d
  ones) stay static; the rest are MLIR `?`, read at run time from `tensor.dim` of a
  host tensor argument or from a runtime scalar argument.
- A full slice (`x[tm, :]`) of a runtime-sized dim is a runtime-sized tile (no
  padding); tiled dims keep their static block size and are padded and masked at
  the end of the loop as for ragged static shapes.
- Runtime sizes the kernel assumes equal (one symbol) or computes (`n // 2`) are
  checked on each call.
- The optimizing pipeline is only used when every linalg op of the inlined module
  is statically shaped (a tiled matmul) and no `linalg.batch_matmul` has padded
  operands (section 14); otherwise the kernel falls back to the scalar pipeline
  (debug log). Lighthouse vectorizes without vector sizes, so it cannot vectorize
  ops on runtime-sized tiles.
- A size no host tensor argument or runtime scalar provides raises a
  `DynamicShapeError`.
- `execute_mlir` (no host code) cannot create host tensors of runtime shape.

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
- Any ATen op without a direct lowering becomes a torch-mlir helper typed by
  its operands at the call site, so support follows torch-mlir's Torch-to-Linalg
  coverage. An op torch-mlir cannot lower fails with an
  `UnsupportedOperationError` at its kernel source line.
- Downstream, lighthouse must still validate and lower the result.

Examples validated in current tests include:
- Elementwise and activation kernels (including runtime `float` parameters,
  e.g. `torch.clamp_max(x, alpha)`), reductions, softmax, layer/RMS norm.
- Nested tiled matmul accumulation patterns.
- Two-operand `torch.einsum` contractions (see below).
- Gathers: one 1-D index tensor per load (`weight[idx[tile_b], tile_e]`), or any
  index tensor into a 1-D tensor. Stores indexed by a tensor (scatter) are
  rejected.

Write a cumulative sum as `torch.cumsum(x, dim)` (Helion replaces it with
`hl.cumsum`). The method form `x.cumsum(dim)` reaches Inductor's CPU lowering
inside Helion's frontend, which fails before the backend runs
(`InductorLoweringError: 'NullHandler' object does not support the context
manager protocol`).

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

## 5) Loop-Carried Values

Every variable a nested loop (or `while`) assigns is carried through its `scf.for`
(`scf.while`), matched to the variable through Helion's `_phi(before, getitem(loop, i))`;
loop inputs that are only read (e.g. a tile loaded before the loop) are not carried. Any
update form works (`acc + x`, `torch.maximum(acc, x)`, `addmm`/`baddbmm`, several
carried values as in online softmax or attention). Lighthouse bufferizes carried values
with `allow-return-allocs-from-loops` (section 15), and `in_place.py` rewrites common
updates to write into the carried buffer.

## 6) Result Shapes Come From the Lowered Operands

Current behavior:
- ATen helpers, `view`/`reshape` and `sym_size` take their shapes from the MLIR
  types of the lowered operands; Helion's symbolic node metadata is never
  modified and is used only for dtypes, ranks and symbol origins.

Consequence:
- A tile's static extent is its block size, or the loop's length when that is
  smaller (one tile then covers the loop exactly, unlike Helion's padded block).
  The block-size symbol (`tile_m` in `hl.zeros([tile_m])` or `x.view(tile_m, -1)`)
  has that same value.

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

## 9) Ragged (Boundary) Tiles

Current behavior:
- A tile whose loop or tensor ends inside it is partial: loads read its real
  part and zero-pad the rest (`tensor.pad`), `_mask_to` sets the padding to the
  reduction identity, and stores write only the real part. This matches
  Helion's masked loads and stores. `extra_mask` zeroes loaded elements and
  skips stored ones.
- A load past the end of a tensor smaller than the iteration domain reads zeros.
- A zero-padded load stored whole into a full destination slice
  (`out[tn.id, tk.id, :, :] = b[tk, tn]` over a padded extent) is written
  directly: a static copy for tiles the source covers, masked row reads plus
  zero rows for edge tiles. See `docs/PADDING_FUSION_FINDINGS.md` (Status) and
  `examples/block_packing_mlir.py`.
- When every block size divides its loop and the loops stay inside the tensors,
  the IR has only static sizes.

Limits:
- A tile offset that may point before the start of a tensor (`x[tile.index - 1]`
  style negative offsets from the first tile) is rejected.
- A padded load transformed before its store (`b_t[tn, tk].permute(1, 0)`), or
  stored into a tile the loop end cuts short, keeps its `tensor.pad`. The result
  is correct but slower.

## 10) Multi-Output Kernels

Current behavior:
- Every host tensor the kernel stores into is an inout argument of the entry
  function and is written in place. Each is threaded as its own SSA value, so
  outputs need not share shapes, and independent top-level loops with different
  grids in one phase are separate `scf.forall`s emitted in sequence.
- Direct calls and `compile_mlir` return the kernel's own `return` value (a
  tuple for `return a, b`).

Consequence:
- `execute_mlir` (no host code) returns every written tensor in entry argument
  order: one tensor, or a `list` when there are several.

## 11) Host Code, Multi-Phase Kernels (`hl.barrier()`) and Host Tensors

Current behavior:
- One module per config: one private function per `hl.barrier()` phase plus a
  memref-ABI entry (see `docs/MLIR_DESIGN.md`, Calling Convention). Direct calls
  and `compile_mlir` run the kernel's host code on every call, before and after
  the device loops, so host-computed tensors (`scale = x.mean()`), module
  globals, runtime scalar parameters, host initialization (`torch.full_like`),
  in-place updates, `out=` parameters and any `return` expression behave as in
  Helion.

Limits:
- `execute_mlir(module, *tensor_params)` runs no host code: host-created tensors
  the kernel writes start zeroed, and kernels that read host-computed tensors or
  take runtime scalar parameters are rejected with an `UnsupportedOperationError`
  naming `compile_mlir`/direct calls.
- Strided (non-contiguous) tensors are copied to contiguous buffers (and copied
  back for written ones); the entry uses identity layouts.
- An input that shares memory with a written tensor is snapshotted (cloned) at
  the call; two written tensors that share memory are rejected.
- Written tensors are identified by their host expression: two host names for
  one storage (a host-side view of a written tensor) are separate arguments and
  are rejected by the overlap check. Read-only reshapes of a declared parameter
  lower to a reshape of it; other read-only host views (`x[:, 2:]`, `x[::2]`,
  `x.t()`) are inputs of their own, computed by the host code (so `execute_mlir`
  rejects them).
- A host tensor whose shape is computed from block sizes
  (`torch.zeros((m, n // block_n))` with `block_n = hl.register_block_size(n)`)
  takes the config's block sizes, as the host code does; a shape depending on
  other runtime sizes is `?` (see Dynamic Shapes).
- In host code, `hl.specialize` and `hl.register_tunable` become their
  compile-time values; other Helion API calls there are rejected.
- No statement other than `hl.barrier()` may appear between two top-level device
  loops (a Helion frontend rule, not backend-specific).

Example:
- `examples/multi_phase_mlir.py` -- a runnable two-phase kernel combining
  `hl.barrier()` with a host-computed interop tensor.

## 12) Backend Architecture Boundary

The MLIR backend is intentionally decoupled from `TritonBackend`. It inherits
from Helion's backend-neutral `Backend` and bypasses Helion's Python AST codegen
and Triton cache-management path. The implementation is split into:


- `lowering/` for operation and control-flow emission.
- `aten_bridge/` for the torch-mlir helpers of generic ATen ops.
- `analysis/` for read-only analyses of Helion's device IR.
- `support/` for index resolution, type conversion and diagnostics.

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

A value of the same rank with size 1 where the slice is wider
(`out[tm, tn] = x[tm, None]`) is broadcast into the destination, as `tl.store`
does.

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
through the optimizing pipeline. The scalar pipeline is unaffected. The autotuner does this
itself: a search under the optimizing pipeline only tries tiles of at least 32 (where the
dimension allows); given configs are not changed.

With tiles of at least 32, lighthouse's transforms still reject (a Python `ValueError`:
"Failed to apply named transform sequence", or "mixed static/dynamic
offset/sizes/strides requires explicit result type" from `move_offsets_to_subview`)
many ops over tiles larger than their dimension, over ragged tiles, and over full
slices whose extent is not a multiple of 32 (a 33x65 softmax row, a 64x40 layer norm).

## 15) Lighthouse Pipeline Deviations

The local lighthouse checkout carries these pipeline changes (to be upstreamed):
- `bufferization.yaml` runs one-shot bufferization with `allow-return-allocs-from-loops`, so
  loop-carried values no longer have to be updated in place (previously any such update
  failed with "Yield operand #0 is not equivalent to the corresponding iter bbArg").
- `scalar-lowering.yaml` includes `bufferization-cleanup.yaml` to deallocate the buffers that
  option allows inside loops.
- `move_offsets_to_subview` skips transfers on memrefs with a dynamic shape or dynamic
  strides, whose subview result type the Python helper cannot infer.

The backend's opt pipeline adds two stages from `_compiler/helion_transforms.py`:

- `vectorize_pads`, before the tensor-level vectorization: each statically shaped
  `tensor.pad` becomes a vector read of its source (padded with the pad value) written
  into an empty tensor. Bufferized as is, a pad is a temporary zeroed and then partly
  copied into, and upstream `vectorize_children_and_apply_patterns` (lighthouse's
  `vectorize_all`) forwards reads of that temporary to the copy source with poison
  padding (`LinalgCopyVTRForwardingPattern` ignores the temporary's zeroing): padded
  reductions and `batch_matmul`s of padded tiles gave garbage or NaNs.
- `split_transfers`, after bufferization and before OpenMP (the split needs an
  allocation scope around each transfer): tile extents are runtime values, so every
  tile's vector transfer may be out of bounds and lowers to masked accesses. Upstream's
  full/partial split (`vector.split_transfer_full_partial`) guards each with an
  in-bounds check, so only edge tiles take the masked path (padded bf16 packing was
  2-4x slower without it). Upstream loops forever on a rank-reducing transfer (vector
  rank below the memref's: it creates the check, then fails), so these first get
  leading unit dims. Transfers with a permutation map (transposed tiles) are not split.

## 16) Autotuning

- Only `block_sizes` are tuned (the backend accepts `block_sizes` and `mlir_pipeline`
  config keys; others raise `InvalidConfig`). The pipeline is not searched: a search uses
  `HELION_MLIR_PIPELINE`'s pipeline.
- Candidates run in the tuning process (no precompile subprocess), timed by wall clock.
  A candidate the backend or lighthouse rejects with an error is skipped, but a native
  abort inside MLIR (e.g. the small-tile assertion above) ends the process.
- Compiling on the optimizing pipeline is much slower than on the scalar one for wide
  tiles (a 32x1024 softmax: about 2.5 s of lighthouse passes and 7 s of LLVM JIT), which
  bounds how many configs a search can afford.
- Compiled entries are cached in-process only; lighthouse's `Runner` cannot load a dumped
  object file, so there is no on-disk cache of compiled code (best configs are cached on
  disk by Helion's autotune cache).

## Out of Scope for This Backend Today

- Dynamic-shape kernels on the optimizing pipeline beyond statically tiled ones.
- GPU runtime execution path parity with CPU path in this backend.
- Guaranteed support for all ATen programs independent of pattern shape.

## Related Docs

- `docs/MLIR_USAGE.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `docs/MLIR_DESIGN.md`
