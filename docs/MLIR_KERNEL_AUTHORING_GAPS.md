# MLIR backend: kernel authoring gaps

Findings from writing a high-performance bf16 4K matmul (`helion_matmul_bf16.py`)
against the `mlir` backend with `HELION_MLIR_PIPELINE=1` on an AMX-capable
Xeon (Emerald Rapids). Everything below was reproduced on that setup; each item
states the observed symptom and, where known, the root cause.

> **Status update.** Blockers 1, 2, 4, 5, 6 and 7 are now resolved in
> `helion_mlir_backend`; each is annotated **RESOLVED** below with what was done.
> Blocker 3 (deeper reduction cache tiles) remains open and is a compiler-side
> issue, as are the codegen quality items. Blocker 8 is a Helion frontend limit.

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

### 8. Helion itself rejects 4D matmul

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

Items 1, 2, 4, 5, 6 and 7 are implemented; item 3 is the remaining compiler-side
blocker and items 8 and 9 remain open.

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
8. **`view` / `reshape` on host tensors inside the kernel** so a packed tensor
   can be re-interpreted without a separate host-side pass. *Still open.*
9. **4D matmul in Helion itself** (frontend limitation, see item 8 above).
   *Still open.*

## Re-verification after the fixes

Re-ran the reproducers against the updated backend while attempting to write a
packed/blocked bf16 matmul. Result: **block packing is still not expressible**,
and end-to-end matmul performance is unchanged (peak 935 GFLOP/s at 8 threads,
same as before; `torch.mm` ~3.9 TFLOP/s).

| Case | Before | Now |
| --- | --- | --- |
| Equal tile sizes `[64, 64, 32]`, `[32, 128, 32]` | silently wrong | **correct** |
| `hl.grid` batched matmul, no K loop | `ValueNotFoundError` | builds; fails JIT via blocker 3 |
| Scalar block index + tiled K reduction | n/a | **MLIR assertion abort** |
| Transposed RHS in a contraction | inlining failure | builds; **numerically wrong** |
| `.to(torch.bfloat16)` epilogue | `func.call` type mismatch | builds; **fails JIT** |
| Nested `hl.grid` | n/a | `ValueNotFoundError: node: u5` |

### A. Scalar block index combined with a tiled K reduction — hard crash

This is *the* pattern a blocked matmul needs: a unit-step loop over block
indices with an inner reduction over `K`.

```python
for i in hl.grid(nb):
    acc = hl.zeros([bm, bn], dtype=torch.float32)
    for tk in hl.tile(kk):
        acc = torch.addmm(acc, a[i, :, tk], b[i, tk, :])
    out[i, :, :] = acc
```

```
mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp:3083:
SubViewOp::inferResultType: Assertion `staticSizes.size() == rank &&
"staticSizes length mismatch"' failed.
```

It aborts the process rather than raising. A rank-reducing slice mixing one
scalar index with one tile index appears to build a `subview` whose static-size
list is the reduced rank while the source rank is unreduced.

#### Reproducer

Run with `HELION_MLIR_PIPELINE=1`. The first snippet has no matmul and fails
earlier with a readable verifier error, so it is the better one to fix against;
the second is the same root cause surfacing as an abort after bufferization.

```python
# repro_a1.py -- scalar grid index + tiled slice, copy only
from __future__ import annotations
import torch, helion, helion.language as hl
import helion_mlir_backend  # noqa: F401

@helion.kernel(static_shapes=True, backend="mlir",
               config=helion.Config(block_sizes=[32]))
def copy_slices(a):
    nb, m, kk = a.shape
    out = torch.empty_like(a)
    for i in hl.grid(nb):
        for tk in hl.tile(kk):
            out[i, :, tk] = a[i, :, tk]
    return out

copy_slices(torch.randn(2, 64, 128).to(torch.bfloat16))
```

```
RuntimeError: MLIR inlining failed: Failure while executing pass pipeline:
error: unknown: expected 3 offset values, got 2
 note: see current operation: "tensor.parallel_insert_slice"(%8, %arg2, %arg1, %3)
   <{operandSegmentSizes = array<i32: 1, 1, 2, 0, 0>,
     static_offsets = array<i64: -9223372036854775808, -9223372036854775808>,
     static_sizes = array<i64: 1, 128>,
     static_strides = array<i64: 1, 1>}>
   : (tensor<1x128xbf16>, tensor<2x64x128xbf16>, index, index) -> ()
```

The destination is rank 3 but the insert carries only 2 offsets/sizes/strides.
The scalar-index dimension is dropped from the slice metadata instead of being
kept as an offset with size 1, so the rank-reduced tile is never expanded back
to the destination rank.

```python
# repro_a2.py -- same shape, through a contraction; aborts instead of erroring
from __future__ import annotations
import torch, helion, helion.language as hl
import helion_mlir_backend  # noqa: F401

@helion.kernel(static_shapes=True, backend="mlir",
               config=helion.Config(block_sizes=[32]))
def batched_mm(a, b):
    nb, m, kk = a.shape
    n = b.shape[2]
    out = torch.empty((nb, m, n), dtype=torch.float32, device=a.device)
    for i in hl.grid(nb):
        acc = hl.zeros([m, n], dtype=torch.float32)
        for tk in hl.tile(kk):
            acc = torch.addmm(acc, a[i, :, tk], b[i, tk, :])
        out[i, :, :] = acc
    return out

batched_mm(torch.randn(2, 64, 128).to(torch.bfloat16),
           torch.randn(2, 128, 64).to(torch.bfloat16))
```

```
mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp:3083:
SubViewOp::inferResultType: Assertion `staticSizes.size() == rank &&
"staticSizes length mismatch"' failed.
```

Replacing `hl.grid` with `ti.begin` from an outer `hl.tile(..., block_size=1)`
reproduces both.

Variants that use a
4D packed layout instead fail earlier:

- `a[i, tk, :, :]` (grid index + tile index on a rank-4 tensor) →
  `ArrayRef<AffineExpr>::operator[]: Assertion 'Index < Length'` abort.
- `a[i, tk.begin, :, :]` → builds, but the K loop is **silently dropped**: the
  emitted IR contains a single `extract_slice` at offset 0 and no `scf.for`, so
  the kernel would compute only the first K block.
- `for kb in hl.grid(nbk)` nested inside `for i, j in hl.grid(...)` →
  `ValueNotFoundError: Value not found for node: u5`. Nested grid loops resolve
  the outer indices but not the inner one.

### B. Transposed RHS is numerically wrong

```python
acc = torch.addmm(acc, x[tm, tk], yt[tn, tk].permute(1, 0))
```

now compiles, but the result is NaN in 96.6% of elements. Since the transpose
is folded into `linalg.contract` indexing maps, the maps are likely permuted on
the wrong operand or the accumulator map is not updated to match.

### C. bf16 epilogue still cannot be compiled

```python
out = torch.empty((m, n), dtype=torch.bfloat16, ...)
out[tile_m, tile_n] = acc.to(torch.bfloat16)
```

The stale-helper-shape error is gone, but the kernel now fails at JIT with
`LLVM Translation failed for operation: omp.wsloop` / `omp.parallel`. Reproduced
with `[64, 4096, 32]` and `[128, 1024, 32]`.

### D. Block packing is expressible but not usable

Exactly one packing shape compiles: a whole-panel copy indexed by an `hl.grid`
scalar (`helion_block_pack.py`).

```python
for j in hl.grid(panels):
    out[j, :, :] = source[:, j, :]
```

Tiling inside the grid body (`for j in hl.grid(...): for tk in hl.tile(...)`)
aborts with case A, and tiling all three dimensions of the 3D view
(`for tj, tk, tn in hl.tile([...])`) also aborts, so parallelism can only come
from the panel count and the per-panel copy cannot be blocked or vectorized by
the author.

The result is far too slow to be worth using: ~10 GB/s at 512x512 with 8
threads, and at 4096x4096 a single pack does not complete in a reasonable time
at all — versus the ~1-2 ms a 64 MB read+write repack should take. Packing that
costs more than the matmul defeats the purpose, so the packing path is a dead
end until case A is fixed and the copy can be tiled.

### Revised priorities1. **Fix A** — scalar block index plus a tiled reduction. Without it, `hl.grid`
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
