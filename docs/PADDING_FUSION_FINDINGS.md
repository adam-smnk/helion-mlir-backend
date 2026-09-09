# Padding-Fusion Into Packing Kernels: Findings and Future Work

This document records why fusing zero-padding directly into the AMX packing
kernels (`_pack_a_kernel`, `_pack_b_kernel`, `_pack_a_kernel_t`,
`_pack_b_kernel_t` in `AI-bench/backends/utils/helion_mlir_cpu_utils/matmul.py`)
is currently blocked in the backend, not in kernel authoring. Two independent
root causes were found; both are reproducible with the small scripts below,
copied here as starting points for a future fix.

Context: today, packing for irregular (non-32-divisible) shapes pads via a
host-level `torch.zeros(...)` + slice-assign *before* the packing kernel runs
(see `matmul.py`), which touches the operand's data twice (once to copy into
the padded buffer, once to pack from it). The goal explored here was to fold
the zero-fill and the real-data copy into the same kernel that does the
packing, so padding-needed shapes get the same single-pass treatment as the
already-fast aligned-shape path (see `docs/AMX_MATMUL_OPTIMIZATION_FINDINGS.md`
for that unrelated packing-loop speedup, which *is* shipped).

Reproduce with:

```bash
# scalar pipeline -- all repros below pass here
env -u HELION_MLIR_PIPELINE OMP_NUM_THREADS=4 LD_PRELOAD=/lib64/libtcmalloc.so uv run python temp/<script>.py
# AMX pipeline -- repros below fail here
OMP_NUM_THREADS=4 HELION_MLIR_PIPELINE=1 LD_PRELOAD=/lib64/libtcmalloc.so uv run python temp/<script>.py
```

---

## 1) Multi-Phase Padding+Pack Kernel: Wrong Results Under the AMX Pipeline

**Status:** blocked in the AMX-optimizing pipeline (lighthouse's register-tiling
schedule), not in Helion frontend codegen.
**Probes:** `temp/test_pack_a_fused_padding2.py` (3-phase single kernel, fails
standalone -- no other kernel needed to reproduce),
`temp/test_pack_a_split_kernels.py` (2 separate kernels called back to back,
same failure).

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
`tile_and_fuse.py[gen=tile_and_fuse_annotated]` stage, see Finding 2's IR dump
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
kernel. That framing was incorrect -- see Finding 2 below, which shows the
"unrelated trivial kernel" crashes **on its own**, standalone, for a
completely different and unrelated reason. The two bugs are independent;
this document originally conflated them.

### Potential fixes to investigate

- Compare the bufferized/lowered IR for the crashing multi-phase-final-stage
  case against the correct standalone case (both listed as probes above) at
  each stage of `pipeline.yaml`, the same way Finding 2's IR-dump reproducer
  does, to find exactly which stage in `pipeline.yaml` first diverges/goes
  wrong for the multi-phase case.
- Look at `tile_and_fuse.py[gen=assign_and_propagate_tile_sizes]` and
  `tile_and_fuse.py[gen=tile_and_fuse_annotated]` (the two custom schedules in
  `pipeline.yaml` responsible for register tiling) for any assumption that a
  phase's `scf.forall`'s `shared_outs` init always comes from a fresh
  `tensor.empty()` rather than a threaded-through buffer from an earlier
  phase.
- Bisect by reducing the padding kernel to just 2 phases (zero-fill + pack,
  no separate real-data-copy phase) to see if the bug needs specifically 3
  phases, or reproduces with 2.

---

## 2) Single-Phase Kernel: Boundary-Tile Masking Clamps Instead of Zero-Filling

**Status:** blocked by how this backend lowers out-of-bounds/masked tile
reads; a more fundamental limitation than (1), and would block a single-phase
fused kernel even if (1) were fixed.
**Probes:** `temp/test_auto_masked_read.py` (2D combined tile, direct OOB
read), `temp/test_auto_masked_read_1d.py` (1D single-dim tile, direct OOB
read), `temp/test_local_partial_write.py` (functional
`torch.nn.functional.pad` composition).

### Summary

Three single-phase (no `hl.barrier()`, no multi-kernel) designs were tried, to

**Status:** blocked by how this backend lowers out-of-bounds/masked tile
reads; a more fundamental limitation than (1), and would block a single-phase
fused kernel even if (1) were fixed.
**Probes:** `temp/test_auto_masked_read.py` (2D combined tile, direct OOB
read), `temp/test_auto_masked_read_1d.py` (1D single-dim tile, direct OOB
read), `temp/test_local_partial_write.py` (functional
`torch.nn.functional.pad` composition).

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

Isolated with a minimal 1D case (`test_auto_masked_read_1d.py`): tiling a
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

**Status:** a separate, tangential bug found while investigating Finding 1 by
testing whether an unrelated kernel compiled beforehand could affect the pack
kernel's compilation. It turned out the "unrelated kernel" itself crashes
under the AMX pipeline, **standalone**, with no other kernel involved --
correcting an earlier (wrong) version of this document's Finding 1, which had
misattributed this crash to "cross-kernel state corruption".

**Probes:** `temp/repro_trivial_alone.py` (minimal, standalone repro),
`temp/repro_amx_state_corruption_dump_ir.py` (same repro, but monkey-patches
`lighthouse.pipeline.stage.PassStage.apply`/`TransformStage.apply` to dump the
input IR of every pipeline stage to `temp/ir_dump/stage_NN_*.mlir` --
including an `fsync()` after every write -- *before* that stage runs, so the
crashing stage's input IR survives on disk even though the crash is a native
`SIGABRT` that Python cannot catch).

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

Running `temp/repro_amx_state_corruption_dump_ir.py` dumps 14 stages
(`stage_00` .. `stage_13`) to `temp/ir_dump/` before the process aborts; no
`stage_14` file is written, so `stage_13_TransformStage.mlir` is the input IR
to the crashing stage. Per `temp/ir_dump/stage_log.txt`, that crashing stage
is the last transform in `pipeline.yaml`'s register-tiling flow --
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

...applied to this payload IR (`stage_13_TransformStage.mlir`, the trivial
kernel's own IR after the earlier register-tiling/fusion-root-assignment
stages have already run):

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

Given both findings are backend-level (not kernel-authoring) issues -- one in
lighthouse's register-tiling schedule for multi-phase kernels, one in this
backend's tile-boundary masking semantics -- padding-fusion into the packing
kernels is **not** attempted in the shipped `matmul.py`. The current
host-side `torch.zeros(...)` + slice-assign approach for the padding-needed
case is kept as-is; it is correct and only adds overhead for the relatively
rare non-32-divisible-shape case. The already-shipped nested-tile
packing-loop speedup (see `docs/AMX_MATMUL_OPTIMIZATION_FINDINGS.md` and
`tests/test_mlir_execution.py::TestPaddedPackingAndMultiPhaseExecution`) is
unaffected by either finding above, since it only exercises a single kernel,
single phase, with tile domains matching the (already block-aligned) packed
buffer's own shape. Finding 3 is unrelated to padding-fusion but is recorded
here since it surfaced during this investigation and blocks anyone trying to
run *any* two AMX-piped kernels in one process today.
