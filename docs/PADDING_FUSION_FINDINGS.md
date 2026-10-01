# Padding-Fusion Into Packing Kernels: Findings and Future Work

This document records the work on fusing zero-padding into the AMX packing
kernels (`_pack_a_kernel`, `_pack_b_kernel`, `_pack_a_kernel_t`,
`_pack_b_kernel_t` in `AI-bench/backends/utils/helion_mlir_cpu_utils/matmul.py`).
The Status section is current; Findings 1 to 3 are the original investigation,
kept for history.

Context: packing for irregular (non-32-divisible) shapes in `matmul.py` pads via
a host-level `torch.zeros(...)` + slice-assign *before* the packing kernel runs,
which touches the operand's data twice (once to copy into the padded buffer,
once to pack from it). The goal is to fold the zero-fill and the real-data copy
into the kernel that does the packing, so padding-needed shapes get the same
single-pass treatment as the aligned-shape path (see
`docs/AMX_MATMUL_OPTIMIZATION_FINDINGS.md` for the packing-loop speedup).

## Status: padding fused into single-phase packing kernels

Padding now fuses into one single-phase kernel under both pipelines, with no
host padding and no barrier. The kernel loops over the packed blocks and loads
past the end of the operand, which reads zeros:

```python
for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
    out[tn.id, tk.id, :, :] = b[tk, tn]
```

`examples/block_packing_mlir.py` has the four layouts (`pack_a`, `pack_b`,
`pack_a_t`, `pack_b_t`). It compares them with eager PyTorch, the host-padded
kernel and an inline `linalg.pack`. Three backend changes made this work:

- **The opt pipeline's wrong results.** The partial load is a `tensor.pad`,
  which bufferizes to a temporary that is zeroed, partly copied into, and then
  read. Lighthouse's `vectorization.py[gen=vectorize_all]` (upstream
  `vectorize_children_and_apply_patterns`, its `LinalgCopyVTRForwardingPattern`)
  forwards the read to the copy source with poison padding, ignoring the
  zeroing, so the padded rows hold garbage. Padded reductions and the padded
  `batch_matmul` NaNs were the same bug.
  - Fix: the opt pipeline's `vectorize_pads` stage (`_compiler/helion_transforms.py`)
    rewrites each static `tensor.pad` into a vector read of its source padded with
    the pad value, before bufferization, so no such temporary exists for any consumer.
  - Speed: tile extents are runtime values, so that read may be out of bounds for
    every tile and lowers to masked loads. The pipeline's `split_transfers` stage
    (upstream `vector.split_transfer_full_partial`) adds an in-bounds fast path,
    so only edge tiles are masked. f32 padded packing matches a hand-written
    backend fast path; bf16 edge tiles stay slower on CPUs without AVX512_BF16,
    where LLVM scalarizes masked bf16 loads.
- **`tile.id` store indices.** A store indexed by `tile.id` is now an owned
  (parallel) dimension, so the block loop is an `scf.forall` and no longer a
  sequential `scf.for`.
- **Pack ops.** Both pipelines now run lighthouse's
  `x86/pack_lowering.py[gen=lower_packs_unpacks]` first, so `linalg.pack` and
  `linalg.unpack` (for example from an inline MLIR snippet) are lowered.

Remaining limits:

- The transposed kernels (`pack_a_t`, `pack_b_t`) pad before `linalg.transpose`;
  their padded reads have a permutation map, which the split skips, so every
  tile is masked. For small padded operands they are slower than eager.

Sections 1 to 3 below are the original findings; their probe scripts are no
longer kept.

---

## 1) Multi-Phase Padding+Pack Kernel: Wrong Results Under the AMX Pipeline

**Status:** superseded by the single-phase kernel above (not re-tested). The
garbage in the padded region matches the `vectorize_pads` bug described in
the Status section, not the tiling assumption suspected below.

### Summary

A 3-phase kernel (phase 0: zero-fill a padded buffer via `hl.zeros()`; phase 1,
after `hl.barrier()`: overwrite the real `[0:m, 0:k]` region; phase 2, after a
second `hl.barrier()`: pack the now-fully-populated buffer) compiles and, under
the **scalar** pipeline, produces byte-for-byte correct results. Under the
**AMX** pipeline (`HELION_MLIR_PIPELINE=1`) it produces wrong results (NaN /
garbage) in the padded region -- **reproducible standalone**, as the only
kernel compiled in a fresh process (no other kernel involved).

Splitting the same logic into two independently-decorated `@helion.kernel`
functions (one 2-phase padding kernel, one single-phase packing kernel, called
back-to-back from plain Python) does **not** help -- still wrong under AMX.

### Root cause

The pre-lowering MLIR for the packing phase/kernel was dumped and diffed
against the *exact same logic* compiled as a **standalone** single-phase
kernel (no padding, no barrier): the IR is byte-identical (module-hash-suffix
aside) once you account for buffer identity, and both give correct results
under the scalar pipeline. Only when this same packing logic is the *last*
phase of a barrier-separated, multi-phase kernel does the **AMX** pipeline's
register-tiling/tile-and-fuse schedule (`pipeline.yaml`'s
`tile_and_fuse.py[gen=tile_and_fuse_annotated]` stage, see Finding 3's IR dump
below for exactly which schedule this is) produce wrong results for that
phase. The suspicion is that lighthouse's tile-and-fuse transform makes an
incorrect assumption about the *shared_outs*/output-buffer-identity produced
by a preceding barrier-separated phase (as opposed to a phase's own freshly
materialized `tensor.empty()`), causing it to fuse/tile incorrectly around
the buffer boundary -- but this needs confirmation from someone with deeper
familiarity with lighthouse's tiling transform than was available for this
investigation.

Note: an earlier version of this document (before this update) additionally
attributed a *crash* (not wrong-results) symptom to "cross-kernel state
corruption" from compiling an unrelated trivial kernel before the pack
kernel. That framing was incorrect -- see Finding 3 below, which shows the
"unrelated trivial kernel" crashes **on its own**, standalone, for a
completely different and unrelated reason. The two bugs are independent;
this document originally conflated them.

---

## 2) Single-Phase Kernel: Boundary-Tile Masking Clamps Instead of Zero-Filling

**Status:** resolved. Loads now read only the part of a
tile inside the loop and the tensor and zero-pad the rest, and the ragged
combined-tile rejection is gone. The reproducer below returns zeros past the
input (`tests/test_ragged_tiles.py::test_read_past_tensor_end_is_zero`), so
design 1 works. The analysis below is kept for history.

### Summary

Three single-phase (no `hl.barrier()`, no multi-kernel) designs were tried, to
see if fusion could be done without touching the phase-boundary machinery
implicated in Finding 1 at all:

1. Tile directly over the **padded** domain and index the smaller real tensor
   directly (`a[tile_m, tile_k]`), relying on Helion's own tile-boundary
   masking to supply zero for out-of-range positions.
2. Build a per-iteration local `hl.zeros(...)` tile and overwrite a
   compile-time-constant-sized prefix slice of it with real data
   (`tile_local[:rm, :] = a[...]`).
3. Build the padded local value functionally, via
   `torch.nn.functional.pad(a[...], (0, 0, 0, 32 - rm))`, avoiding any
   subscript-assignment into a local device value.

Design 2 is rejected outright at trace time:

```text
helion.exc.DeviceTensorSubscriptAssignmentNotAllowed: Cannot assign to subscript of device tensor 'tile_local'.
```

(`torch.cat` was also tried as a alternative functional composition and is
rejected differently, at Inductor lowering time: `InductorLoweringError:
Lowering aten.cat.default returned buffer type <class
'torch._inductor.ir.ConcatKernel'>, expected ComputedBuffer` -- concat isn't
lowerable to a single `ComputedBuffer` on this backend's Inductor-lowering
path.)

Designs 1 and 3 both **compile** but silently produce **wrong data** (not
zero) in the out-of-bounds region.

### Root cause

Isolated with a minimal 1D case: tiling a
64-wide output in two blocks of 32 while reading from a 19-element input, the
**first** block (containing real data, indices 0-18, and some in-block
padding, indices 19-31) computes correctly. The **second** block (indices
32-63, entirely beyond the input's real 19-element extent) does **not** read
as zero -- it duplicates the first block's real data instead.

This shows that this backend's boundary-tile masking is a **safety clamp**
(preventing an actual out-of-bounds memory access by clamping the read index
back into the tensor's valid range), not a **semantic zero-fill** (substituting
a default value like `0.0` for out-of-range elements, as e.g. Triton's masked
loads with `other=0.0` do). Because the clamp happens before any per-element
computation, wrapping it in `torch.nn.functional.pad` (design 3) does not
help -- the pad op's own zero-fill logic never sees an genuinely out-of-range
read to begin with; the clamp has already substituted real data for it.

This is consistent with (and now explains) a limitation already visible
elsewhere in this backend's error messages: `"ragged combined-tile block
size ... this backend does not yet support a dynamically-sized boundary tile
in this position"` (see `helion_mlir_backend/_compiler/mlir/lowering/control_flow.py`).
Any single-phase kernel that tiles over a domain larger than an input
tensor's real extent will read incorrect (duplicated/clamped) data for the
excess region, independent of the multi-kernel/multi-phase issue in Finding 1.

### Reproducer (minimal, 1D)

```python
import torch
import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401


@helion.kernel(static_shapes=True, backend="mlir", config=helion.Config(block_sizes=[32]))
def _read_oob_1d_test(a: torch.Tensor, n_pad: hl.constexpr) -> torch.Tensor:
    out = torch.empty((int(n_pad),), dtype=a.dtype, device=a.device)
    for t in hl.tile(int(n_pad)):
        out[t] = a[t]
    return out


torch.manual_seed(0)
a = torch.randn(19)
res = _read_oob_1d_test(a, hl.constexpr(64))
ref = torch.zeros(64)
ref[:19] = a
print("correct:", torch.equal(res, ref))     # False under HELION_MLIR_PIPELINE=1
print("res[19:64] should be 0, is:", res[19:64])  # shows duplicated a[0:...] data, not zeros
```

### Potential fixes to investigate

- The fix belongs in this backend's index/mask lowering (likely
  `helion_mlir_backend/_compiler/mlir/support/index_meta.py` and/or
  `lowering/tile_index_ops.py`, wherever a tile's read gets clamped to a
  tensor's real extent): when a load's computed index range falls (partially
  or fully) outside the *source* tensor's shape, and the read is not being
  used to compute a store destination, it should materialize a genuine
  `linalg.fill`-zero (or an `arith.select`-based masked load) for the
  out-of-range elements instead of clamping the index.
- Alternatively, expose a masked-load primitive (mirroring `hl.load`'s
  `extra_mask=` parameter in Helion core, see
  `helion/helion/language/memory_ops.py`) that this backend actually lowers
  to a real zero-filled masked load, rather than the current clamp-based
  masking used for boundary safety.
- Any fix here should be validated against the *existing* general
  "independent top-level loops with incompatible geometry" and "ragged
  combined-tile block size" `UnsupportedOperationError`s in
  `control_flow.py`, since those checks currently exist specifically because
  boundary reads are unsafe today; a correct zero-fill fix might allow
  relaxing or removing some of those checks.

---

## 3) Unrelated: AMX Tile-And-Fuse Schedule Crashes on a Standalone 1D Elementwise Kernel

**Status:** current, and not specific to padding: the optimizing pipeline aborts on
ops whose tiled dims are all smaller than 32 (`docs/MLIR_LIMITATIONS.md`, section 14;
`scripts/lighthouse_small_tile_repro.py`). It was found while investigating Finding 1,
whose earlier version misattributed this crash to "cross-kernel state corruption".

### Summary

```python
import torch
import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401


@helion.kernel(static_shapes=True, backend="mlir", config=helion.Config(block_sizes=[8]))
def _trivial_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * 2.0
    return out


x = torch.randn(64)
r = _trivial_kernel(x)  # crashes under HELION_MLIR_PIPELINE=1, standalone
print("trivial kernel alone ran fine:", torch.equal(r, x * 2.0))
```

Under `HELION_MLIR_PIPELINE=1` this crashes with:

```text
mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp:710: LogicalResult applyTilingToAll(RewriterBase &, Operation *, Range &&, unsigned int, transform::TransformResults &, bool, function_ref<FailureOr<scf::SCFTileAndFuseResult> (TilingInterface)>):
Assertion `tiledResults->loops.size() == numLoops && "Mismatched number of loops, tile and fuse transform should have failed"' failed.
```

Under the scalar pipeline it runs fine and gives correct results.

### IR at the point of the crash

Dumping the input IR of every pipeline stage shows the crashing stage is the
last transform in `pipeline.yaml`'s register-tiling flow --
`tile_and_fuse.py[gen=tile_and_fuse_annotated]`:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = "transform_ext.get_fusion_roots"(%0) : (!transform.any_op) -> !transform.any_op
    transform.foreach %1 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %2 = "transform_ext.get_tile_sizes"(%arg1) : (!transform.any_op) -> !transform.any_param
      %transformed, %loops = transform.structured.fuse %arg1 tile_sizes *(%2) {apply_cleanup, use_forall} : (!transform.any_op, !transform.any_param) -> (!transform.any_op, !transform.any_op)
      %3 = "transform_ext.clear_tile_and_fuse_annotations"(%loops) : (!transform.any_op) -> !transform.any_op
    }
    transform.apply_cse to %arg0 : !transform.any_op
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
```

...applied to this payload IR (the trivial kernel's own IR after the earlier
register-tiling/fusion-root-assignment stages have already run):

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @_trivial_kernel(%arg0: memref<64xf32>, %arg1: memref<64xf32>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.to_tensor %arg1 restrict : memref<64xf32> to tensor<64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = scf.forall (%arg2) = (0) to (64) step (8) shared_outs(%arg3 = %1) -> (tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %0[%arg2] [8] [1] : tensor<64xf32> to tensor<8xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.elementwise kind=#linalg.elementwise_kind<mul> indexing_maps = [#map, #map1, #map] {transform_ext.tile_sizes = array<i64: 0>} ins(%extracted_slice, %cst : tensor<8xf32>, f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg3[%arg2] [8] [1] : tensor<8xf32> into tensor<64xf32>
      }
    }
    bufferization.materialize_in_destination %2 in restrict writable %arg0 : (tensor<64xf32>, memref<64xf32>) -> ()
    return
  }
}
```

Note the `linalg.elementwise` op is already annotated
`{transform_ext.tile_sizes = array<i64: 0>}` (zero tile size) from an earlier
stage (`tile_and_fuse.py[gen=assign_elementwise_tile_sizes]`), and is already
sitting inside a hand-rolled `scf.forall` (from Helion's own codegen, not
lighthouse tiling) with no further un-tiled dimension left to fuse. The
`get_fusion_roots`/`transform.structured.fuse ... tile_sizes *(%2)` transform
apparently still attempts to tile-and-fuse this op with 0 requested loops
against an op that (for this reason, or another not yet identified) doesn't
produce the expected number of loops, tripping the C++-side sanity assertion
in `applyTilingToAll` (upstream MLIR, not lighthouse or this backend).

### Potential fixes to investigate

- Reproduce the exact IR above directly in `mlir-opt`/`mlir-transform-opt`
  (or equivalent) outside Python entirely, using the two MLIR snippets dumped
  above, to get a completely minimal, backend-independent repro for upstream
  or lighthouse.
- Check `transform_ext.get_fusion_roots`/`get_tile_sizes` (see
  `lighthouse/lighthouse/dialects/transform/transform_ext/`) for whether they
  should be excluding ops already annotated with a zero tile size
  (`transform_ext.tile_sizes = array<i64: 0>`) from the fusion-roots set
  entirely, rather than handing them to `transform.structured.fuse` anyway.
- Try the same trivial kernel with a block size that evenly divides 64
  without a fractional last tile (e.g. `block_sizes=[16]` or `[32]`) to see
  if the crash is specific to this shape/block-size combination or general to
  any single-level `hl.tile()`-only elementwise kernel under the AMX
  pipeline.

---

## Recommendation

Fused padding now works in one single-phase kernel on both pipelines (Status
above), but the shipped `matmul.py` still pads on the host. Switching its
packing kernels to the fused form of `examples/block_packing_mlir.py` would
remove the extra pass over the operand for padding-needed shapes. The transposed
layouts would first need the padded transposed read to be split like the others
(see Remaining limits).
