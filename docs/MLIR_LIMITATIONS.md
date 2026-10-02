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
- The optimizing pipeline vectorizes ops on runtime-sized tiles with masks (section 15).
  A runtime extent that tiling leaves without a constant bound is tiled by 32 first;
  an op that cannot be tiled (e.g. a `keepdim` reduction) is lowered to scalar loops.
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
- On the optimizing pipeline a zero-padded load is a vector read of its real
  part, padded past its end, and only edge tiles take the masked path (section 15;
  `examples/block_packing_mlir.py`).
- When every block size divides its loop and the loops stay inside the tensors,
  the IR has only static sizes.

Limits:
- A tile offset that may point before the start of a tensor (`x[tile.index - 1]`
  style negative offsets from the first tile) is rejected.
- A padded tile read transposed (`b_t[tn, tk].permute(1, 0)`) keeps a permuted
  vector read, which the optimizing pipeline does not split: every such tile takes
  the masked path, which is slower (much slower for bf16 on CPUs without
  AVX512_BF16, where LLVM scalarizes masked bf16 loads).

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

## 14) Small Tiles on the Optimizing Pipeline

Lighthouse's cache-level tile-and-fuse assigns a zero tile size to every dimension smaller
than its 32-element cache tile. Upstream MLIR aborts the process when it fuses an op whose
tile sizes are all zero (`applyTilingToAll`: "Mismatched number of loops");
`scripts/lighthouse_small_tile_repro.py N` reproduces it with a single
`linalg.elementwise` on `tensor<Nxf32>` (N < 32 aborts). The local lighthouse change of
section 15 skips such fusion roots, so small tiles work on the optimizing pipeline.

An autotuning search under the optimizing pipeline tries tiles of at least 32 (where the
dimension allows) except for the leading dim of each outermost loop, which may be any
size: small outer tiles add parallel work and keep the code per tile small, while
narrower inner tiles waste vector lanes. Given configs are not changed. On lighthouse's
main branch, a small outer tile of a kernel without another tiled dim still aborts.

## 15) Lighthouse Pipeline Deviations

The backend relies on two lighthouse changes not yet on its main branch:
- `bufferization.yaml` runs one-shot bufferization with `allow-return-allocs-from-loops`,
  so loop-carried values need not be updated in place (otherwise such updates fail with
  "Yield operand #0 is not equivalent to the corresponding iter bbArg"; section 5).
- `get_fusion_roots` skips ops whose tile sizes are all zero (section 14).

The optimizing pipeline (`_compiler/pipeline.yaml`, the default) follows lighthouse's x86
pipeline, with these stages from `_compiler/helion_transforms.py` replacing or added to it:
- `pin_transposes`, before the cache-level tiling: a static transpose of at most 4096
  elements with no linalg producer or user (a tile moved to another layout, such as
  `b[tk, tn].reshape(16, 2, 32).permute(0, 2, 1)`) is annotated with zero tile sizes,
  which lighthouse's tiling keeps, so it becomes one vector transpose. Tiled, rank-3
  transposes were split per leading index and unrolled per element of a narrow inner
  dim, and a padded load fused into the split took a heap-allocated temporary per
  slice (VNNI packing in `examples/vnni_packing_mlir.py` was up to 2.5x slower than
  plain block packing; it now takes about as long).
- `vectorize_pads`, before the tensor-level vectorization: each `tensor.pad` whose
  runtime extents have evident constant bounds becomes a vector read of its source
  (padded with the pad value) written into an empty tensor. Bufferized as is, a pad is a
  temporary zeroed and then partly copied into, and upstream `vectorize_children_and_apply_patterns`
  (lighthouse's `vectorize_all`) forwards reads of that temporary to the copy source with
  poison padding (`LinalgCopyVTRForwardingPattern` ignores the temporary's zeroing):
  padded reductions and `batch_matmul`s of padded tiles gave garbage or NaNs.
- `vectorize_linalg` (lighthouse's, from `x86_64/vectorize.yaml`) vectorizes ops of
  runtime shape with masks. Lighthouse vectorizes without vector sizes, which fails for
  them. Tiling bounds a runtime extent by an `affine.min` with a constant; the bounds,
  traced through slices, pads and loops, are the vector sizes. Other runtime extents
  are tiled by 32 first, and so are bounds above 32 that are not multiples of 32. All
  extents above 32 of any op, static or not, whose vectors would exceed 4096 elements
  are tiled by 32 too (LLVM otherwise spends seconds to minutes on them: a 32-row
  softmax tile of 1024 columns compiled in about 110 s, now about 1 s). A masked
  add-contraction becomes an unmasked one of
  operands zeroed where masked off (upstream's x86 contraction patterns rewrite inside
  `vector.mask` regions, which the verifier rejects). Ops that cannot be tiled or that
  vectorization rejects (e.g. argmax, gathers) are lowered to loops
  (`convert-linalg-to-loops`). Loop-free (0-d) ops are not vectorized: upstream turns
  one reading with `tensor.extract` into invalid IR.
- `split_transfers`, after bufferization: tile extents are runtime values, so every
  tile's vector transfer may be out of bounds and lowers to masked accesses. Each is
  split on an in-bounds check into an in-bounds transfer and the original, so only edge
  tiles take the masked path (padded bf16 packing was 2-4x slower without it). Upstream's
  `vector.split_transfer_full_partial` is not used: it stages n-D vectors through a
  stack buffer with `vector.type_cast`, which overflows the buffer when the inner
  vector dim is not a power of two in bytes (LLVM pads each inner vector), and it loops
  forever on rank-reducing transfers. Transfers with a permutation map (transposed
  tiles) are not split.
- `legalize_for_llvm`, before the LLVM lowering, rewrites what upstream's lowering
  rejects or gets wrong. 0-d transfers (lowered upstream only on memrefs of unit inner
  stride) and i1 transfers (LLVM packs an i1 vector into bits, while a memref holds
  one byte per i1: bool tensors read and wrote garbage) become per-element scalar loads
  and stores. Contractions with operands narrower than the accumulator (bf16 operands
  folded into an f32 contraction for the x86 dot-product and AMX patterns) that no x86
  pattern took get their operands extended again; without AMX or AVX512-BF16 they
  otherwise fail to lower.

## 16) Autotuning

- Only `block_sizes` are tuned (the backend accepts `block_sizes` and `mlir_pipeline`
  config keys; others raise `InvalidConfig`). The pipeline is not searched: a search uses
  the default pipeline (`HELION_MLIR_PIPELINE`, else `opt`).
- Candidates run in the tuning process (no precompile subprocess), timed by wall clock.
  A candidate the backend or lighthouse rejects with an error is skipped, and so is one
  whose lighthouse lowering runs past the compile timeout or aborts natively (lowering
  runs in a forked process). A native abort in the LLVM JIT still ends the process.
- Compiling on the optimizing pipeline is slower than on the scalar one (a 256x1024
  softmax: about 1 s for any row block size). Tiles of several rows over very wide rows
  are the exception: 2-row tiles of 16384-column rows take about 4 minutes, nearly all in
  one-shot bufferization's analysis of lighthouse's unrolled register tiles, while
  1-row tiles compile in about a second.
- Compiled entries are cached in-process only; lighthouse's `Runner` cannot load a dumped
  object file, so there is no on-disk cache of compiled code (best configs are cached on
  disk by Helion's autotune cache).

## Out of Scope for This Backend Today

- GPU runtime execution path parity with CPU path in this backend.
- Guaranteed support for all ATen programs independent of pattern shape.

## Related Docs

- `docs/MLIR_USAGE.md`
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md`
- `docs/MLIR_DESIGN.md`
