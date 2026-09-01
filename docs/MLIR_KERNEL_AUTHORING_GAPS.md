# MLIR backend: kernel authoring gaps

Findings from writing a high-performance bf16 4K matmul (`helion_matmul_bf16.py`)
against the `mlir` backend with `HELION_MLIR_PIPELINE=1` on an AMX-capable
Xeon (Emerald Rapids). Everything below was reproduced on that setup; each item
states the observed symptom and, where known, the root cause.

> **Status update.** Scalar grid/tile indexing, nested grid indexing, equal tile
> sizes, dtype epilogues, transposed RHS contractions, mixed-precision
> contractions, and static view/reshape lowering are covered by the backend
> regression suite. Deeper reduction cache tiles (`TK > 32`) remain open as a
> compiler-side AMX issue, as do the codegen quality items. 4D matmul remains a
> Helion frontend limitation.

## Summary

The backend already lowers the canonical Helion matmul idiom
(`hl.tile` + `hl.zeros` + `torch.addmm`) into `linalg.matmul ins(bf16, bf16)
outs(f32)` and lighthouse turns that into `tdpbf16ps`. That path works and is
numerically excellent (max abs error ~1.8e-4 against an f32 reference at
K=4096).

What is missing is everything needed to *control layout and loop structure*.
A kernel author can currently express exactly one shape of matmul: a
two-level `hl.tile([m, n])` / `hl.tile(k)` nest over row-major 2D tensors, with
`K`-cache-tile pinned to 32. Every technique that a CPU matmul normally relies
on — block packing, explicit multi-level tiling, transposed operands, deeper
`K` accumulation in registers — is unavailable, and the resulting code runs at a
small fraction of AMX peak.

## Blockers

### 1. `hl.grid` is not supported — blocks all packed/blocked layouts

```python
for i in hl.grid(nb):
    out[i, :, :] = torch.matmul(a[i, :, :], b[i, :, :])
```

fails with `ValueNotFoundError: Value not found for node: u1`.

The device graph is clean and small:

```
%u1  = _get_symnode(u1)
%load = load(%a, [%u1, slice(None), slice(None)], ...)
```

`_lower_get_symnode` in `_compiler/mlir/codegen.py` only understands
`block_size_N` keys; it calls `block_id_from_key(key)` and raises when the key
is a grid-index symbol. `grid_block_ids` is already `[[0]]` and the enclosing
`scf.forall` induction variable is already in `ctx.block_id_to_iv`, so the
missing piece is mapping the symbol to that IV, plus rank-reducing
`tensor.extract_slice` / `parallel_insert_slice` for scalar index positions.

**This is the single highest-value fix.** Unit-step grid loops over outer block
dimensions with 2D views inside are how blocked matmul is written in plain MLIR,
and they are the only way to feed the contraction pre-packed, contiguous tiles.

**RESOLVED.** Grid symbols are resolved through `HostFunction.expr_to_origin`
(`GridOrigin`) rather than the `block_size_N` key, and loads/stores now emit
rank-reducing `tensor.extract_slice` / rank-expanding
`tensor.parallel_insert_slice` for scalar-indexed dimensions. `a[i, :, :]`
yields a genuine 2D tile and lowers to `linalg.matmul`.

### 2. `tile.begin` is not supported

```python
a[ti.begin, tk.begin, :, :]
```

raises `UnsupportedOperationError: Unsupported operation: tile_begin`. This is
the other route to scalar block indices, so it is blocked for the same reasons
as `hl.grid`.

**RESOLVED.** `tile.begin`, `tile.end`, `tile.id`, `tile.count` and
`tile.block_size` all lower to scalar `index` arithmetic off the enclosing loop
induction variable. `tile.end` clamps with `arith.minsi` only when the extent is
not an exact multiple of the block size.

### 3. Reduction cache tile is pinned to the AMX register tile

Any `block_sizes=[TM, TN, TK]` with `TK != 32` for bf16 fails at JIT time:

```
error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
RuntimeError: JIT compilation failed: Failure while creating the ExecutionEngine.
```

Verified failing for `TK` in {64, 512, 2048}; only `TK == 32` compiles.

Cause: with `TK > 32` the `register_reduction` strategy tiles `K` again, and
`hoisting.py[gen=hoist_loops]` turns the accumulator into vector-typed
`iter_args`. The AMX conversion needs a memory-backed accumulator, so
`vector.contract` survives to LLVM translation.

Consequence: the f32 accumulator is re-loaded and re-stored from memory every
32 elements of `K`. For a 4096³ matmul that is ~1 KiB of accumulator traffic per
output element (16 GB total), which dominates runtime.

The scalar-grid plus tiled-K lowering path itself is covered for the full-slice
contraction form. Source-rank static sizes for scalar-plus-tiled slices are now
correctly emitted and are covered by IR regression tests. Native execution of
some scalar-grid plus tiled-slice forms still has a separate runtime/bufferization
failure and remains open, independently of the deeper AMX reduction/cache tile
conversion for `TK > 32`.

### 4. Non-distinct tile sizes silently produce wrong results

`block_sizes=[32, 32, 32]` and `[32, 64, 32]` compile and run, but the output is
wrong (`Tensor-likes are not close!`). No diagnostic is emitted.

Cause: `lowering/control_flow.py::lower_nested_for_loop` recovers the block id of
a nested loop by searching `block_id_to_size` for a block whose size divides the
upper bound. Equal sizes make that lookup ambiguous and it picks the wrong one.

At minimum this should be a hard error. Ideally the block id should be carried
explicitly rather than re-derived from sizes.

**RESOLVED.** The true cause was in index lowering, not the loop: a
`sym_size.int` index node loses its symbolic metadata when
`_restore_placeholder_metadata` concretizes placeholder shapes, so the block id
fell back to size matching, which is ambiguous when sizes are equal. The block
id of each placeholder dimension is now recorded while the symbols are still
live and used directly, so equal tile sizes resolve exactly. A hard error also
guards the remaining size-matching fallback.

### 5. dtype-cast epilogue fails

```python
out = torch.empty((m, n), dtype=torch.bfloat16, ...)
...
out[tile_m, tile_n] = acc.to(torch.bfloat16)   # also fails as an implicit store cast
```

```
'func.call' op operand type mismatch: expected operand type 'tensor<64x64xf32>',
but provided 'tensor<256x512xf32>'
```

The outlined ATen helper is built for a stale default block shape (64x64) and
`_rebuild_aten_helper_for_call` does not recover. This forces f32 outputs and
blocks any mixed-precision epilogue.

**RESOLVED.** `.to(dtype)` (both the `call_method` and the
`aten._to_copy` / `prims.convert_element_type` forms) is lowered directly to an
elementwise `arith` conversion instead of an outlined helper, and a store into a
narrower output inserts the cast implicitly.

### 6. In-kernel transpose / permute of a tile fails

```python
torch.addmm(acc, x[tm, tk], yt[tn, tk].permute(1, 0))
```

fails during `MLIR inlining`. A transposed-B matmul is the cheapest way to make
the RHS AMX tile rows contiguous, and it is not expressible.

**RESOLVED.** A `permute` / `transpose` / `t` that only swaps the two innermost
dimensions and feeds a contraction is folded into `linalg.contract` with
indexing maps encoding the transpose. A standalone transpose lowers to
`linalg.transpose`.

Both the emitted indexing maps and numerical execution against `x @ yt.t()`
are covered by integration and execution tests.

### 7. Mixed-precision contraction takes the fallback path

`lowering/matmul_ops.py::emit_matmul_like` returns `None` when the accumulator
element type differs from the operand element type:

```python
if out_type.element_type != lhs_type.element_type:
    return None
```

bf16 → f32 therefore never uses the direct `linalg.matmul` emitter and instead
goes through `batch_import_and_lower`. It happens to produce the right IR, but
the fast path should accept it explicitly: `linalg.matmul` supports mixed
precision natively and this is *the* shape the AMX strategy looks for
(`is_amx_bf16_contraction` requires bf16/bf16/f32).

**RESOLVED.** `emit_matmul_like` now accepts a wider accumulator than its
operands (bf16/bf16 -> f32, f16/f16 -> f32, i8/i8 -> i32) and keeps emitting
`linalg.matmul` / `linalg.batch_matmul` directly.

### 8. Host-tensor view / reshape and 4D matmul

Static shape-changing `view` / `reshape` operations are now lowered through
`tensor.reshape`-compatible MLIR paths and covered by a regression test. Dynamic
or tile-proxy reshapes remain constrained by Helion's frontend rules.

### 9. Helion itself rejects 4D matmul

`torch.matmul with input tensor dim <2 or >3 is not supported in Helion kernel`.
Not a backend issue, but it means blocked layouts must go through item 1
(scalar block indices + 2D views) rather than through rank-4 ops.

## Codegen quality issues (compiler side, not authoring)

These are not authoring blockers but they cap achievable performance, and the
kernel author has no lever to work around them:

- **RHS VNNI repack is inside the innermost loop.** Every 16x32 B tile is
  rebuilt with a 16-iteration shuffle loop into a `memref.alloca` before
  `x86.amx.tile_load`. The LHS and the accumulator load directly from subviews;
  only the RHS goes through scratch. With no way to pre-pack B (items 1, 2, 6)
  this repack is redundant across the M loop.
- **Accumulator is spilled per AMX tile.** `x86.amx.tile_store` writes to a
  `memref<16x16xf32>` alloca which is then copied out with
  `vector.transfer_read` / `transfer_write`, instead of storing to the
  accumulator subview directly.
- **Nested OpenMP regions.** `omp.py[gen=parallelize]` converts every
  `scf.forall`, so the register-tile forall inside each cache tile becomes an
  `omp.parallel` nested in the cache-tile `omp.parallel`, once per K step.

Net effect measured with 8 threads: ~0.9 TFLOP/s for the Helion kernel versus
~3.5 TFLOP/s for `torch.mm` on the same tensors, and only ~4% of AMX peak.
Cache-tile shape barely matters (514–936 GFLOP/s across a wide sweep), which is
consistent with the bottleneck being the innermost loop rather than the blocking.

## Requested support, in priority order

Items 1, 2, 4, 5, 6, 7 and static item 8 support are implemented; item 3 is the
remaining compiler-side blocker. Item 9 remains a Helion frontend limitation.

1. ~~**`hl.grid` / scalar block indices**~~ — done, including rank-reducing load
   and store (`a[i, kb, :, :]`, `out[i, j, :, :] = acc`).
2. ~~**`tile.begin`**~~ — done, along with `tile.end`, `tile.id`, `tile.count`
   and `tile.block_size`.
3. **Deeper reduction cache tiles** (`TK > 32`), i.e. keep the register-level
   `K` loop accumulator memory-backed so AMX conversion still applies.
   *Still open — compiler side.*
4. ~~**Hard error on ambiguous block sizes**~~ — done, and the distinctness
   requirement is removed: equal tile sizes now lower correctly.
5. ~~**dtype conversion ops in the epilogue**~~ — done, including implicit cast
   on store.
6. ~~**Transpose / permute of a loaded tile**~~ — done via `linalg.contract`
   indexing maps, with `linalg.transpose` for standalone uses.
7. ~~**Direct mixed-precision path in `emit_matmul_like`**~~ — done.
8. ~~**Static `view` / `reshape` on host tensors inside the kernel**~~ — done
   for statically shaped forms; dynamic and tile-proxy forms remain frontend
   constrained.
9. **4D matmul in Helion itself** (frontend limitation).
   *Still open.*

## Re-verification after the fixes

Re-ran the reproducers against the updated backend while attempting to write a
packed/blocked bf16 matmul. Result: **block packing is still not expressible**,
and end-to-end matmul performance is unchanged (peak 935 GFLOP/s at 8 threads,
same as before; `torch.mm` ~3.9 TFLOP/s).

| Case | Before | Now |
| --- | --- | --- |
| Equal tile sizes `[64, 64, 32]`, `[32, 128, 32]` | silently wrong | **correct** |
| `hl.grid` batched matmul, no K loop | `ValueNotFoundError` | **correct and numerically tested** |
| Scalar block index + tiled K reduction | n/a | full-slice form **correct and verifier-tested**; scalar-plus-tiled native execution remains open |
| Transposed RHS in a contraction | inlining failure | **correct and numerically tested** |
| `.to(torch.bfloat16)` epilogue | `func.call` type mismatch | **correct and numerically tested** |
| Nested `hl.grid` | `ValueNotFoundError: node: u5` | **correct and verifier-tested** |

### A. Scalar block index combined with a tiled K reduction — RESOLVED

This is *the* pattern a blocked matmul needs: a unit-step loop over block
indices with an inner reduction over `K`.

```python
for i in hl.grid(nb):
    acc = hl.zeros([bm, bn], dtype=torch.float32)
    for tk in hl.tile(kk):
        acc = torch.addmm(acc, a[i, :, tk], b[i, tk, :])
    out[i, :, :] = acc
```

Nested loop block IDs and source-rank slice metadata are now corrected. The
full-slice contraction form lowers and verifies successfully, and the expected
`1x64x32` source-rank metadata is regression-tested. Mixed scalar-grid and
trailing tiled-slice native execution still needs a runtime/bufferization fix.

### B. Transposed RHS is numerically wrong — RESOLVED

```python
acc = torch.addmm(acc, x[tm, tk], yt[tn, tk].permute(1, 0))
```

The case is now numerically correct. The transpose is folded into
`linalg.contract` indexing maps, with execution verified against `x @ yt.t()`.
The emitted indexing maps and numerical result each have regression coverage.

### C. bf16 epilogue — RESOLVED

```python
out = torch.empty((m, n), dtype=torch.bfloat16, ...)
out[tile_m, tile_n] = acc.to(torch.bfloat16)
```

The stale-helper-shape and JIT failures are fixed for the supported tested path.
Explicit narrowing casts emit `arith.truncf`, widening casts emit `arith.extf`,
and implicit stores into bf16 outputs cast the stored tile. IR and numerical
execution regression tests cover these cases.

### D. Block packing — RESOLVED for nested panel/tile copies

The packing kernels now use a nested panel-grid plus inner tile loop. This
avoids one giant whole-panel extract/insert operation and exposes bounded
vector-shaped copies to the MLIR pipeline:

```python
for panel in hl.grid(panel_count):
   for tile_k in hl.tile(k):
      out[panel, tile_k, :] = source[tile_k, panel, :]
```

The A-panel kernel uses the analogous M-tile form. Both kernels are numerically
covered by execution tests against `permute(...).contiguous()` references.

```python
for j in hl.grid(panels):
    out[j, :, :] = source[:, j, :]
```

The former whole-panel and fully multi-dimensional variants were retried:

```python
for j in hl.grid(nbn):
   for kk in hl.grid(k):
      for nn in hl.grid(bn):
         out[j, kk, nn] = source[kk, j, nn]
```

fails during nested loop lowering because all nested grid dimensions have block
size 1 (`Ambiguous block id for nested loop: block ids [1, 2] all have size 1`).

```python
for panel in hl.grid(nbn):
   for tile_k, tile_n in hl.tile([k, bn]):
      out[panel, tile_k, tile_n] = source[tile_k, panel, tile_n]
```

fails during module creation (`Check that kernel has static_shapes=True and all
ops are in hl.tile() loops`). A pure 3D tiled transpose:

```python
for tile_panel, tile_k, tile_n in hl.tile([nbn, k, bn]):
   out[tile_panel, tile_k, tile_n] = source[tile_k, tile_panel, tile_n]
```

segfaults (`status=139`) at 512x512. Pure 4D tiled layouts also abort in MLIR
with allocator corruption (`malloc(): unsorted double linked list corrupted`):

```python
for tile_panel, tile_k_block, tile_k, tile_n in hl.tile([nbs, kbs, bk, bn]):
   out[tile_panel, tile_k_block, tile_k, tile_n] = \
      source[tile_k_block, tile_k, tile_panel, tile_n]
```

A standalone
`source.permute(1, 0, 2).contiguous()` is rejected by Helion before the backend
(`NoDeviceLoopsInKernel`).

Those variants remain unsupported, but the two-level panel/tile kernels used by
`helion_block_pack.py` are now functional. With `OMP_NUM_THREADS=1`,
`HELION_MLIR_PIPELINE=1`, and the MLIR Python bindings configured, the default
512x512 benchmark completes under 30 seconds and passes its numerical checks.
With `OMP_NUM_THREADS=4`, the 4096x4096 pack also completes under `timeout 30`:
`pack B` is ~2.38 ms and `pack A` is ~2.48 ms, around 27-28 GB/s per operand on
the shared node.

Consuming the packed RHS in a matmul is now numerically correct for exact K-tile
boundaries. The formulation

```python
for j in hl.grid(nbn):
   for tile_m in hl.tile(m):
      acc = hl.zeros([tile_m, bn], dtype=torch.float32)
      for tile_k in hl.tile(k):
         acc = torch.addmm(acc, x[tile_m, tile_k], packed_b[j, tile_k, :])
      out[j, tile_m, :] = acc
```

is covered by the f32 reproducer and execution regression tests. The fix keeps
explicit source tile symbols authoritative when lowering loads inside a
synthetic accumulator store context; previously the A slice used the outer
panel IV for both dimensions, effectively loading `A[panel, panel]` instead of
`A[tile_k, tile_m]`.

Ragged K tails in a tiled contraction remain open: with `K=96` and `TK=64`,
the final iteration currently still emits a fixed 64-wide contraction instead
of a 32-wide tail. The packed-RHS regression therefore covers exact K-tile
sizes, while existing copy tests cover ragged slice metadata separately.

A standalone f32 reproducer is available in `helion_block_packed_f32_repro.py`.
With `SIZE=128`, `BLOCK_M=32`, `BLOCK_N=32`, `BLOCK_K=64`,
`OMP_NUM_THREADS=4`, and `HELION_MLIR_PIPELINE=1`, it now passes both the pack
and packed-matmul numerical checks under `timeout 30`.

The analogous bf16 packed-RHS consumer is still blocked in AMX lowering. At
512x512 with exact `K=512`, `TK=32`, and `BN` in `{32, 64, 128}`, the packed
bf16 consumer fails JIT with:

```
LLVM Translation failed for operation: builtin.unrealized_conversion_cast
```

The non-packed transposed-RHS bf16 tiled form also remains incorrect in the
benchmark pattern (`~96%` mismatched elements at 512x512):

```python
acc = torch.addmm(acc, x[tile_m, tile_k], yt[tile_n, tile_k].permute(1, 0))
```

So the only currently correct bf16 high-performance path is still the row-major
explicit-`K=32` loop in `helion_matmul_bf16.py`.

The no-explicit-K-loop matmul variant was also retried:

```python
for tile_m, tile_n in hl.tile([m, n]):
   out[tile_m, tile_n] = torch.mm(x[tile_m, :], y[:, tile_n])
```

It fails at JIT with the same deeper reduction-cache-tile issue:
`LLVM Translation failed for operation: builtin.unrealized_conversion_cast`.

### E. bf16 AMX optimization: what packing bought, and the remaining ceiling

Packing the RHS into contiguous `[N/BN, K, BN]` panels is worth ~3x, but **not**
because it removes the VNNI repack. The repack is still emitted, still inside the
K loop, and still redundant. What packing changed is the *cost* of that repack:
the row-major kernel gathered `vector<16xbf16>` from a memref with row stride `N`
(32 cache lines, half-used), while the packed kernel reads contiguous
`vector<32xbf16>` from the panel.

The final IR keeps two accumulators in AMX tile registers across K, with the VNNI
conversion in a nested loop:

```
^bb6(%160: i64, %161: !llvm.x86_amx, %162: !llvm.x86_amx):   // K loop
  ^bb8: ... llvm.shufflevector %202, %192 [0, 32, 1, 33, 2, 34, 3, 35, ...]
  ^bb10: 3 x tileloadd64 ... 2 x tdpbf16ps
  llvm.br ^bb6(%267, %261, %266)
```

`tilezero` runs before the loop and `tilestored64` only after it, so the
accumulators do not spill. Note the shuffles are named `llvm.shufflevector` after
LLVM lowering, not `vector.shuffle`; grepping for the MLIR spelling gives a false
negative.

Because the B repack sits inside the `tile_m` loop, each B tile is re-converted
`M / PACK_TILE_M` times. Hoisting it, or accepting a pre-VNNI-packed operand,
would remove that redundancy.

The second limit is instruction-level parallelism: there are only **two**
accumulator chains (`%161`, `%162`), each serially dependent across K.
`tdpbf16ps` has roughly 52-cycle latency against 16-cycle throughput, so at least
four independent accumulators are needed to saturate the unit.

The accumulator count is `(TM/16) * (BN/16)`, and both are pinned:

| variant | result |
| --- | --- |
| `TM = 32` | JIT failure, `vector.contract` survives |
| `TM = 64` | `SubViewOp::inferResultType` assertion abort |
| `BN = 64` / `BN = 128` | assertion abort |
| `TK > 32` (incl. whole-K) | blocker 3, JIT failure |
| bf16 output epilogue | JIT failure |

Thread scaling is essentially linear (336 / 746 / 1438 GFLOP/s at 1 / 2 / 4
threads), so this is latency-bound per core, not bandwidth-bound. M-blocking the
grid to `(m_block, n_panel)` was therefore worth only a few percent, but it is
kept because it raises the parallel work item count from 128 to 1024, which
matters at 64 threads.

### F. Split-K: exact failure mode

Split-K is the natural way to add accumulator chains without raising `TM`/`BN`,
and the shapes it produces are correct. Before the AMX rewrite, the base kernel
has 2 and the split-K kernel has 4 `vector.contract` ops, all of the same
AMX-friendly shape:

```
vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
```

Despite that, split-K fails:

```
error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
error: LLVM Translation failed for operation: omp.wsloop
```

some `vector.contract` ops are simply not converted to AMX.

Two things were ruled out:

- **It is not the final `acc0 + acc1` reduction.** A diagnostic variant that
  keeps both chains but stores only `acc0`, with no vector add consuming the
  accumulators, fails identically.
- **C-tile duplication does not avoid it.** Writing the partials to separate
  output slices (`out[0, ...] = acc0`, `out[1, ...] = acc1`) so the reduction
  happens outside the kernel hits a different backend failure:
  `mlir/lib/IR/PatternMatch.cpp: RewriterBase::eraseOp: Assertion`
  `mayBeGraphRegion(*op->getParentRegion()) && "expected that op has no uses"`.

So the blocker is that the AMX conversion handles only a single contraction chain
per loop body; two independent loop-carried accumulators in one `scf.for` are not
converted, regardless of how they are consumed.

### G. Unpack cannot be written as a Helion kernel

The packed matmul produces `[M/BM, N/BN, BM, BN]`. Converting that to row-major
`[M, N]` needs a `[M/BM, BM, N/BN, BN]` result, which then `view`s to `[M, N]`
for free. Every formulation of that store indexes a rank-4 destination as
`(scalar, tile, scalar, full)`, and all of them fail:

| formulation | result |
| --- | --- |
| matmul writes `out[m_block, tile_m, panel_n, :]` directly | wrong results (97% mismatch), then segfault |
| separate kernel, `hl.grid([mbc, np]) / hl.tile(bm)` | segfault |
| separate kernel, `hl.grid(mbc) / hl.tile([bm, np])` | module creation failure |

Writing the matmul output directly in `[M/BM, BM, N/BN, BN]` order is the most
valuable of these: it would make the unpack a free `view` and remove a whole pass
over the output. Until the interleaved scalar/tile store works, unpack stays an
eager `permute().contiguous()` (~2.3 ms at 4K, 4 threads).

### Revised priorities

1. **`TM > 16` / `BN > 32` register tiles for packed bf16** — currently a JIT
   failure or an MLIR assertion abort. This gates AMX ILP and is the main cap on
   the packed kernel.
2. **Multiple accumulator chains per loop body** (split-K, section F). Either
   path would raise ILP; both are backend failures today.
3. **Hoist the VNNI repack out of the `tile_m` loop**, or accept a pre-packed
   VNNI operand. Today each B tile is re-shuffled `M / PACK_TILE_M` times.
4. **Blocker 3** — deeper reduction cache tiles (`TK > 32`). This remains a
   compiler-side performance limitation and caps AMX utilization.
2. Codegen quality: RHS repacking, accumulator spills, and nested OpenMP
   regions remain performance work.
3. 4D matmul remains a Helion frontend limitation.

<!-- Historical priorities retained below for context. -->
<!-- 1. **Fix A** — scalar block index plus a tiled reduction. Without it, `hl.grid`
   and `tile.begin` cannot be used for anything except a batched matmul with the
   whole `K` in one contraction, which blocker 3 then rejects. Blockers 1 and 2
   are therefore only nominally resolved: no blocked matmul can be written.
2. **Blocker 3** — deeper reduction cache tiles. Unchanged, and it is what caps
   the current kernel at ~4% of AMX peak (accumulator round-trips to cache every
   32 elements of `K`).
3. **Fix B** — the transposed-RHS miscompile is a correctness bug and should
   probably be disabled until fixed.
4. **Fix C** — bf16 epilogue.
5. Nested `hl.grid` index resolution.
-->

---

## Architecture Notes: Descriptor-Driven Slice/Store Lowering (Phases 0-5 Refactor)

**Goal**: Eliminate positional/size heuristics from load/store lowering. Replace with authoritative metadata from Helion index expressions.

**Key Changes**:

1. **Phase 0: `slice_plan.py`** — New descriptor abstraction (`DimSlice`, `SlicePlan`) that captures per-dimension geometry (kind: scalar/tile/full, offset, size, block_id, reduces).

2. **Phase 1: Lower-load descriptor rewrite** — `load_slice_ops.py` now uses `plan_slice()` instead of guessing sizes from `for_store_ctx_stack`, value shape, or matching extents. Fast path (1-D gather) preserved.

3. **Phase 2: Nested loop generalization** — Removed `assert len(block_ids) == 1` from `lower_nested_for_loop`. Divisibility heuristic fallback removed (kept inner block ID resolution from body symbols).

4. **Phase 3: Synthetic store on descriptor** — `memory_ops.py` per-iteration insert uses `plan_slice()` on the synthetic store context directly (the legacy `inner_dim` fallback path was removed after empirical verification — see "Legacy cleanup" below).

5. **Phase 4: Terminal store + grid dims** — `build_kernel_body` derives which grid block_id maps to which output dimension (not positional) from the terminal store's own index expression. Terminal store path tries the descriptor first, then a positional fallback (this fallback is the common case in practice, since the real output tensor usually has no SSA value bound yet at terminal-store time).

6. **Phase 5: Cleanup + hardening** — `infer_block_id_from_value_shape()` (kept as a deprecated fallback in the original pass) was later confirmed dead via empirical instrumentation and removed — see "Legacy cleanup" below.

**Legacy cleanup** (verified via empirical instrumentation: temporarily added stderr markers to every suspected-dead fallback branch, ran the full test suite with `-s`, counted hits, removed anything with zero hits):
- Removed `build_context.py::infer_block_id_from_value_shape` and its 2 call sites — symbolic block-id resolution always suffices; this shape-based guess never fired.
- Removed the legacy `inner_dim`-based fallback in `memory_ops.py::lower_store`'s synthetic-store branch, along with the `try/except` that swallowed `plan_slice` failures to reach it — `plan_slice` never fails there across all tested nesting depths, combined tiles, transposes, and reductions; a real failure now surfaces as a real error.
- Removed the now-write-only `"block_id"`/`"inner_dim"`/`"rank"` keys from `control_flow.py`'s `synthetic_store_ctx` dict, and the dead `BuildContext.block_id_to_out_dim` field (set but never read back).
- Confirmed **still active** (kept): the positional heuristic-based terminal store (the most common path in practice), `_find_reused_block_id`'s recursive wrapper-unwrapping, the `fallback_outer_bid` heuristic, and the loop-declaration-order fallback in `build_kernel_body`.

**Invariants Maintained**:
- Index position == tensor dimension (scalars reduce rank via `reduces=True`)
- Block IDs come from `node_symbol_info()` or `infer_index_block_and_bias()`, never from size/extent matching
- `for_store_ctx_stack` is optional (descriptor path doesn't require it)
- Synthetic store accumulator carry is separate from slice geometry

**Test Coverage**: 128/128 tests pass (`uv run pytest`, `HELION_MLIR_PIPELINE` unset — that env var switches to the AMX vectorizing `pipeline.yaml`, which is for the bf16 matmul benchmark scripts only, not general kernels).

**Post-refactor bug fixes** (found via crash/correctness triage, not part of the original 6 phases):
- `codegen.py` reused a fresh `mlir.ir.Context()` per compile; rapid create/destroy across kernel configs raced the native context's background thread pool and corrupted the heap. Fixed by sharing one process-wide `Context` (standard MLIR usage: one context, many modules).
- `slice_plan.py`'s `plan_slice()` used the absolute block/tile induction variable as the offset even when the base tensor's dimension had already been reduced to one local tile (e.g. a synthetic per-iteration accumulator), writing out of bounds. Fixed: offset is forced to 0 whenever the base extent equals exactly one tile's size.
- `control_flow.py`'s `block_id_to_out_dim` mapping assumed grid-block ids map to output dimensions in loop-declaration order. Two bugs from that assumption: (1) a single `hl.tile([m, n])` statement produces one `grid_block_ids` group containing both block ids, which all collapsed onto one output dimension; (2) the store's index order can differ from loop declaration order (e.g. `out[tm, panel, :]` with the `panel` loop declared first). Fixed by deriving the mapping from the actual terminal store's index expression (authoritative), falling back to flattened loop order only if no matching store is found.

**Arbitrary-depth nested loop generalization** (implements the two previously-open unpack forms):
- `lower_nested_for_loop` no longer asserts a single block id per loop node. It resolves each level's real block id (`_find_reused_block_id` now recurses through pure-wrapper nested `_for_loop` bodies to arbitrary depth, fixing `grid -> grid -> tile` 3+ levels deep) and, for a combined multi-dim tile (`hl.tile([a, b])`, one `_for_loop` node carrying 2+ block ids), disambiguates each dimension via `_resolve_multi_block_ids` (matches declared upper bound against each candidate block's real size hint), then emits one `scf.for` per block id via a shared recursive emitter (`_emit_for_loop_level`).
- Every loop level between the outer `scf.forall` and the level with the actual store now threads its own synthetic accumulator (found via `_find_descendant_store`, a proper DFS scoped to true descendants — not a global scan), chained through `ctx.push_store_ctx`/`ctx.for_store_ctx_stack` exactly like naturally-nested `_for_loop` FX nodes already did. This required two additional fixes surfaced by testing: (1) the legacy `fallback_outer_bid` heuristic could misfire on a resolvable-but-not-yet-active descendant block or a full-slice dim, now guarded to only apply to genuinely unresolvable non-slice indices; (2) the tail flush into a parent's local accumulator now clamps an ancestor's offset to 0 whenever the parent's own dimension has already been reduced to a single slot (mirroring the `slice_plan.py` fix, but for this separate legacy code path).
- `load_slice_ops.py` no longer builds a single rank-reducing `tensor.extract_slice` (ambiguous, and can trigger a native assertion, whenever a *kept* tile dimension also happens to have extent 1 — e.g. a `hl.tile()` with block size 1). It now always extracts at full rank, then explicitly drops only the scalar-indexed dims via `tensor.collapse_shape` with an index-based reassociation map (unambiguous, since dims are named by position, not inferred from size).

**Future Work / remaining gaps**:
- **Unpack operations** — all 3 originally-planned forms now work:
  - `grid(np) → tile(m)`, including reordered store indices (`out[tm, panel, :]`) — covered by `test_unpack_grid_tile_reordered_store_execute_mlir`.
  - `grid(mb) → grid(np) → tile(bm)` (3 levels deep) — covered by `test_unpack_triple_nested_grid_execute_mlir`.
  - `grid(mb) → tile([bm, np])` (grid + a single combined 2D tile) — covered by `test_grid_combined_2d_tile_execute_mlir`.
- **Dimension-reordering transpose in a combined tile — resolved, not a gap**: an *implicit* reorder via differing load/store index order (e.g. `out[m,tm,tp,:] = src[m,tp,tm,:]`, swapping `tm`/`tp`) is genuinely invalid Helion syntax when the swapped dims differ in size — Helion's own frontend type-checks the assigned value's shape against the store's expected shape and would reject a real mismatch (an earlier test of this only "passed" the frontend by degenerate luck, using a block size of 1 for one dim). The *correct*, already-fully-supported way to write this is an **explicit** `.permute()`/`.transpose()`/`.t()` call — Helion's device IR already represents these as standard `aten.permute`/`aten.transpose` ops, and the backend already has a dedicated `linalg.transpose` lowering (`transpose_ops.py`) for them. Investigating this surfaced a real, narrower, pre-existing bug: `aten_lowering.py::_fake_tensor_from_load_node` (used only when building torch-mlir "ATen helper" subgraphs) reconstructed a load's shape by counting index positions **without dropping scalar-indexed (grid/`tile.begin`/literal-int) dimensions**, disagreeing with the load's real (correctly rank-reduced) `meta['val']` — this broke any ATen op consuming a scalar-indexed load's result directly, not just permute in a combined tile. Fixed to drop scalar-indexed dims the same way `plan_slice`/`ctx.is_scalar_index_node` already do elsewhere. Covered by `test_scalar_grid_index_transpose_execute_mlir` (minimal case) and `test_grid_combined_2d_tile_explicit_transpose_execute_mlir` (the original motivating case).
- **Combined multi-dim tile with an external loop-carried accumulator — resolved, not a gap**: verified empirically (device-IR dumps + an executing numerical test) that Helion's device IR *never* attaches a carried accumulator directly to a combined `hl.tile([a, b])`'s own `_for_loop` node — every dimension in a combined tile is parallel by construction, a genuine reduction always gets its own separate single-block `_for_loop` nested inside (already fully supported), and host tensors read inside a combined tile are re-materialized via `_host_tensor`, never lifted as iter-args. The realistic pattern (`grid → tile([m, n]) → tile(k)` batched matmul) already executes correctly and is covered by `test_grid_combined_tile_separate_reduction_execute_mlir`. The `NodeLoweringError` guard for the non-empty-iter-args case is kept as a defensive assertion (documented as unreachable on current Helion IR) rather than removed outright.
- Ragged K-dimension handling (K not exact multiple of TK) remains open as a separate issue.
