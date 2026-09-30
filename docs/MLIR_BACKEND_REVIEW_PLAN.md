# helion_mlir_backend: In-Depth Review and Improvement Plan

Status: proposal for review (revision 2: lighthouse investigation folded in, detailed
per-phase plan in §5). No backend code has changed yet.
Baseline: `uv run pytest -q tests/` gives 284 passed in 77 s. The package has about 8.3k LOC.
Probe and spike scripts that reproduce every claim below are in `temp/review_probes/`. Run
each case in its own process, e.g. `uv run python temp/review_probes/probe.py ragged1d`.

Decisions already agreed:

| Topic | Decision |
|---|---|
| ATen ops | Keep torch-mlir. Build helpers at the call site from real MLIR operand types. Write direct linalg codegen only where torch-mlir is a clear blocker, with a fallback. |
| Custom lowerings | Keep every lowering that works around a torch-mlir or lighthouse shortcoming (§4.3). Consolidate them, but do not generalize them away. |
| Ragged tiles | Keep static tiles. Pad on load, apply a real `_mask_to`, and store only the valid part. Take a static fast path when the block size divides the extent. |
| Calling convention | Follow Helion semantics. Writing into host tensors is OK. The in/out semantics must be explicit in the MLIR itself. |
| Entry function | The backend emits its own memref-ABI entry instead of lighthouse's `result_to_args`, with the reason documented. Everything else reuses lighthouse. |
| Non-contiguous tensors | Make them contiguous in the driver (free when already contiguous). Strided memrefs are untested and deferred. |
| Phases | Keep one function per phase (they may share one module). `hl.barrier()` does nothing on CPU. |
| Outer loops | Emit a normalized `scf.forall` (unit step) so lighthouse's SFC remap applies, without changing Helion semantics. |
| Pipelines | The backend owns its pipeline YAMLs and adds bufferization options as needed. Deviations are listed for later upstreaming (§9). |
| Lighthouse API | Align with the current API as is. No adapter layer. |
| bf16 execution | Not run on this non-AMX machine. AMX is validated at IR level: `x86.amx.*` ops must appear after x86 vectorization. |
| Small tiles | A lighthouse assertion (§9) is fixed upstream. Until then, optimizing-pipeline workloads use tiles ≥ 32. |
| Approach | Incremental. The test suite stays green at every step. |
| Helion upstream | No edits to Helion. Monkeypatching is OK. |

---

## 1. Executive summary

The backend works for the patterns its tests cover. It is brittle outside them: probes of
common Helion idioms return wrong results, crash the heap, or fail with a misleading error.
Almost all of the heuristic complexity, and most of these bugs, trace back to four root causes:

1. **Device IR is misread.** `_for_loop(graph_id, begin, end, args)` is treated as
   `(graph_id, block_ids, upper_bounds, args)`. The real block ids live in
   `device_ir.graphs[graph_id].block_ids` (`ForLoopGraphInfo`). The default `begin=[0]`
   produced the idea that "Helion reuses outer block id 0 as a placeholder". About 250 LOC of
   disambiguation heuristics plus several bug fixes recorded in repo memory exist only because
   of this. A nonzero `begin` is silently ignored, which gives wrong results.
2. **Stores do not follow SSA.** Root-level stores are deferred to the `scf.forall` terminator.
   Stores inside nested loops go into a "synthetic accumulator" whose geometry is guessed from
   the **first** descendant store. The first store's slice plan is then reused for every later
   store. As a result, read-after-store, multiple stores, stores in `if`, and outputs that
   host code pre-initialized are all broken.
3. **ATen helper shapes are guessed before codegen.** They are reconstructed from stale
   symbolic `meta` with fake evaluation, broadcast "normalization" and clipping, then rebuilt
   at the call site when the guess is wrong. That is about 900 LOC. The code also mutates
   Helion's own device-IR metadata in place.
4. **Host semantics are bypassed on the single-phase fast path.** Host statements are not
   executed and outputs are freshly allocated. So in-place kernels, `out=` parameters, partial
   writes into `torch.full_like(...)`, and runtime scalar arguments do not work.

A fifth factor constrains every fix:

5. **Lighthouse only accepts specific IR shapes.** Its GEMM anchors are named contractions or
   `linalg.contract`. AMX needs one bf16×bf16→f32 contraction op. Its bufferization config
   rejects loop-carried values that are not updated in place. Its `result_to_args` cannot
   express in/out tensors. Some of today's custom lowerings exist for exactly these reasons,
   so the plan keeps them (§4.3) and moves the remaining workarounds into backend-owned
   pipeline configuration.

Fixing the four root causes structurally turns most of the heuristic code into small,
authoritative logic. The net deletion should be roughly 2–3k LOC, while every capability
listed in §4.3 is preserved and guarded by tests.

---

## 2. Evidence (probe results)

| Probe | Result | Root cause |
|---|---|---|
| `hl.tile(2, n)` (nonzero begin) | **wrong** (max err 4.1) | §1.1: begin read as block id 2, lower bound forced to 0 |
| 1-D elementwise, extent 100, block 16 | **heap crash** (`double free or corruption`) | out-of-bounds `extract/insert_slice` on the ragged last tile |
| matmul with ragged K (k=20, bk=8) | **NaN** | out-of-bounds read, no masking |
| row max with ragged N | lighthouse failure | same |
| row softmax (`x[tm, :]`, `amax`/`exp`/`sum`) | helper signature mismatch | same-rank broadcast "normalization" turns `8x1` into `8x32` |
| `acc + torch.addmm(b, x, y)` | **wrong** (max err 14) | `aten_target_matches` substring match: `"mm"` matches `addmm` |
| `out[t] = ...; out[t] = out[t] * 2` | error "No value for tensor node out" | outputs have no SSA value inside the body |
| nested-loop store plus outer store to the same output | **wrong** | first-store geometry and reused first `store_plan` |
| `if t.begin == 0:` | unsupported `eq` / `_if` | `_if`, `_and`/`_or`/`_not` not lowered |
| `hl.grid(0, n, 2)` | unsupported `_for_loop_step` | not lowered |
| `torch.where(v > 0, v, zeros_like(v))` | torch-mlir pipeline failure | one bad helper fails the whole batch; no isolation |
| float scalar kernel argument | torch-mlir import failure | SymFloat embedded as a literal. Helion keys floats **by type only**, so baking the value in would be silently wrong on the next call. |
| partial write into `torch.full_like(x, 7)` | **wrong** | host init never runs; fresh `torch.empty` output |
| in-place `x[t] = x[t] + 1` | error "No value for tensor node x" | output removed from the inputs |
| `out=` parameter kernel | **not written** | a fresh buffer is returned; the caller's `out` is untouched |
| several configs on one `BoundKernel` | OK | (metadata mutation is latent, not observed) |
| `int // 3`, `f64` add | OK | |
| non-contiguous input (`x.t()`) | **wrong** (max err 4.29) | the identity-layout ABI ignores the strides that lighthouse's memory manager passes |
| `hl.dot(x, y, acc=acc)` | unsupported `dot` | not lowered |

### 2.1 Lighthouse spikes

These are hand-written MLIR modules in `temp/review_probes/spike_lighthouse.py` and
`spike_sfc.py`. They use tiles of at least 32 unless noted.

| Spike | Scalar pipeline | Optimizing pipeline | Consequence for the plan |
|---|---|---|---|
| In-place update via a backend memref entry (`to_tensor restrict writable` + `materialize_in_destination`) | OK | OK | Phase 4 ABI. No full-buffer copy after bufferization. |
| Returning an argument unchanged through `result_to_args` | error "Unsupported return value" | same | Reason to emit our own entry. |
| Owned-tile state threading (`extract_slice(shared_out)` → `insert_slice` → `parallel_insert_slice`) | OK | OK | Phase 3. Canonicalizes back to today's IR (0 `insert_slice` left). |
| Ragged 1-D: pad on load + partial store | OK | OK | Phase 6. |
| Ragged-K matmul with zero-padded operands | OK | OK | Phase 6. |
| Runtime scalar as a 0-d tensor argument | OK | OK | Phase 4. |
| Phase functions + entry in one module | OK | OK | Phase 4. |
| Unfused `acc = acc + matmul` in `scf.for` | error "Yield operand #0 is not equivalent" | same | Lighthouse bufferization config. |
| Same, with `allow-return-allocs-from-loops` + deallocation | OK | OK | Phase 0: backend-owned pipelines. Full suite 284/284 with the scalar variant. |
| Normalized outer forall (`in (4, 4)` + `affine.apply` offsets) | n/a | SFC remap applied, exact | Phase 1. A stepped forall is never remapped. |
| Any op whose tiled dims are all < 32 | OK | **abort** in `applyTilingToAll` | Upstream bug. Reproducer in §9. |
| bf16 blocked matmul with AMX features forced on | n/a | `x86.amx.tile_load/tile_mulf/tile_store/tile_zero` produced | Phase 0 AMX IR gate. |

torch-mlir, for comparison, lowers these contraction patterns to several ops:
- `addmm` → fill, `matmul` and `generic` add.
- A bf16 `acc + mm` → two cast generics, an f32 `matmul` and an add.
- The blocked `einsum` → three transposes, reshapes, `batch_matmul` and an add.

None of these is a single lighthouse anchor, and the bf16 one never qualifies for AMX.

---

## 3. Findings by area

### 3.1 Device-IR interpretation
- `control_flow.lower_nested_for_loop`, `_find_reused_block_id`, `_resolve_multi_block_ids`,
  `_block_id_matching_extent`, `_resolve_block_upper_bounds` (the "skip grid ids" fix),
  `symbolic_shape_restoration` (reads `node.args[2]` as upper bounds) and the cross-phase
  "stale block id" fix all work around §1.1.
- `block_id_to_upper_bound` is inferred from misread args and output shapes, then merged with
  `min()`. The authoritative extent is `env.block_sizes[bid].size`, together with the loop's
  own `begin`/`end`.
- `_for_loop_step`, `_if`, `_while_loop`, `_constant_tensor`, `_and`/`_or`/`_not`, `hl.dot`,
  `_reduce`, `_associative_scan`, atomics, `split`/`join` and `rand` have no lowering.
  `hl.dot` appears in 12 of Helion's 58 upstream examples, and `torch.where` in 15.
- Silent fallbacks: a dynamic block size becomes `-1` and is then used as a loop step;
  `_lower_sym_size` falls back to `0`; `for_block_id_stack[-1]` is used as a "last resort"
  block id.

### 3.2 Control flow and stores
- One `scf.forall` holds **all** root graphs of a phase. Independent top-level loops with
  different grids are rejected ("incompatible geometry"), although in Helion they are
  independent loops.
- `forall_insert_slices` is an untyped tuple list. `ForStoreContext` and the synthetic
  accumulator logic (about 400 LOC) are a special case of general tensor-state threading.
- `memory_ops.lower_store` has three store paths (synthetic, bound, positional). The
  positional path, which re-derives offsets from the stored value's shape, is the common one.
- `_existing_output_value` finds an output's initial value by scanning `meta['val'] is tensor`
  across graphs.
- `OutputTensorResolver` guesses outputs with multi-step precedence and fallbacks, including
  "the last parameter".

### 3.3 ATen bridge
- The pre-pass guesses shapes (`_fake_tensor_from_node_meta`, `_resolve_dims[_or_none]`,
  broadcast normalization, `_compute_conservative_common_shape`, `refresh_aten_tensor_meta`).
  `helper_rebuild` then fixes the guess at the call site with a second torch-mlir run that
  temporarily mutates `node.meta`.
- The code writes real `torch.zeros(...)` tensors into Helion's shared `node.meta['val']`.
  This is a latent cross-config hazard and costs memory (a full `x[tm, :]` tile per node).
- The torch-mlir pipeline runs over the whole batch, so one failing op fails every helper.
- Name matching: 34 sites use `target.__name__`/`str(target)` substring checks. Confirmed bugs:
  `mm`→`addmm`, and `div.Tensor` matching `div.Tensor_mode` (which ignores `rounding_mode`).
  `_normalize_aten_args` rewrites `mul(x, None)` into `mul(x, x)`, which is a suspicious hack.
- Casts, gathers and `tile.index` use `tensor.generate` plus `tensor.extract`. This is
  scalarized IR that `linalg-fuse-elementwise-ops` and vectorization cannot use.

### 3.4 Structure and code reuse
- About 20 codegen methods in `codegen.py` are one-line wrappers
  (`_lower_x → lowering.lower_x(self.context, ...)`). `node_dispatch.py` holds a dict of
  lambdas. `lower_custom_aten` is a hand-ordered chain of `if aten_target_matches(...)`.
- There are three contraction paths (`emit_matmul_like`, `_emit_contract`, `lower_einsum`) and
  two "acc + X" fusions. All of them are one concept: a contraction with an optional
  accumulator and folded transposes.
- `BuildContext` mixes immutable geometry, per-function SSA state, loop stacks, store stacks,
  helper maps and a callback.
- 248 function-local imports, a defensive pattern kept from optional-dependency concerns.
  Identical snippets (constant, fill, empty-plus-zero, reshape via `from_elements`) repeat 5+
  times.
- Dead code: `_process_root_graphs`, `_process_graph`, `infer_index_block_and_bias`, the
  `"zeros"` dispatch entry (`hl.zeros` lowers to `full`), and stale `__pycache__` from old
  module names.

### 3.5 Runtime and host semantics
- There are two runtime paths: a single-phase fast path and a multi-phase host-prefix driver.
  They diverge in semantics.
- `run()` drops non-tensor args, and an executor allocates outputs itself.
- `api.generate_mlir` mutates the user's `kernel.settings.backend`.
- The pipeline is selected through the global env var `HELION_MLIR_PIPELINE`.
  `MLIRBackend.autotune` always returns the default config.
- `emit_matmul_like` with no accumulator accumulates in the operand dtype (bf16), so its
  numerics differ from torch.

### 3.6 Diagnostics
- Custom errors are not tied to the user's source. Every device-IR node has
  `meta["location"]`, and wrapping each node's lowering in `with node.meta["location"]:` would
  point errors at the kernel line.
- `ModuleBuilderError("module_creation", hint="Check static_shapes=True ...")` wraps
  unrelated failures with a misleading hint. 24 `except Exception` blocks are too broad.

### 3.7 Tests
- The suite is good and fast (284 tests / 77 s) with property tests.
- There is no `conftest.py` harness (`_execute` is duplicated), no differential run against
  Helion's upstream examples, no ragged, in-place or scalar-argument cases, and no crash
  isolation. One heap corruption takes down the whole run.

### 3.8 Lighthouse interaction
- **Anchors.** `tile_and_fuse` anchors on `linalg.matmul`, `matmul_transpose_a/b`,
  `batch_matmul`, `batch_reduce_matmul`, `linalg.contract` and the matvec family. A
  contraction expressed as `linalg.generic` is not an anchor.
- **AMX.** `is_amx_bf16_contraction` needs one contraction op with bf16 lhs/rhs and an f32
  accumulator, all rank ≥ 2. This is why the fused `acc + einsum/mm` lowering matters.
- **Loop-carried values.** `bufferization.yaml` runs one-shot bufferization without
  `allow-return-allocs-from-loops`. Every `scf.for` yield must therefore be an in-place update
  of its iter arg. Today only `matmul(outs=acc)` satisfies that. Under the optimizing pipeline
  even a DPS copy fails, because tile-and-fuse rewrites the loop body.
- **`result_to_args`.** It is tensor-only, marks every input `restrict` (so a buffer cannot be
  both input and output), and rejects a function that returns an argument unchanged.
- **Strides.** `TorchMemoryManager` passes torch strides, but the pipelines bufferize function
  boundaries with an identity layout, so strides are ignored.
- **Small tiles.** The optimizing pipeline's cache tiling zeroes tile sizes for dims below 32.
  An op with all-zero tile sizes then aborts in upstream `applyTilingToAll`. This happens for
  any payload, Helion or not.
- **SFC remap.** `sfc_remap_forall` only rewrites 2-D foralls with `lb = 0`, `step = 1`, static
  bounds, and both IVs feeding contraction slice offsets. Helion's stepped forall never
  qualifies.
- **Parallelism.** `omp.py` turns every `scf.forall` into OpenMP, including Helion's outer
  loop. Nested foralls from cache tiling become nested parallel regions.
- **Known miscompiles/blockers**, recorded in `docs/AMX_MATMUL_OPTIMIZATION_FINDINGS.md`:
  - A transposed inner-block B is silently wrong under AMX.
  - Rank-5 VNNI contractions fail to vectorize.
  - Outer tiles larger than `[1, 1]` hit an `eraseOp` assertion.

  The plan must not emit these shapes by default, and must not claim to fix them.

### 3.9 Strengths to keep
`index_meta.resolve_index_descriptor` (authoritative indexing), `SlicePlan`, einsum capture and
`einsum_spec`, the shared MLIR context, `host_prefix`, careful docstrings, and the empirical
dead-code audits.

---

## 4. Target architecture

### 4.1 Layout (reached incrementally; names chosen to minimize churn)

```
helion_mlir_backend/
  __init__.py  api.py  inject.py              # public API + monkeypatches (unchanged role)
  _compiler/
    backend.py                                # MLIRBackend
    driver.py                                 # compile_config: host prefix -> entry fn (was bound_kernel.py)
    runtime/  executor.py  pipelines/{scalar,opt}.yaml  # lighthouse lowering + JIT (was execution.py)
    analysis/                                 # read-only, no MLIR
      geometry.py        # KernelGeometry: per block id size/extent/begin/step, loop tree
      tensor_effects.py  # host tensors read/written per graph (transitive), store ownership
      signature.py       # KernelSignature: tensor roles (in/inout), scalars, per-phase IO
      host_prefix.py
      canonicalize.py    # FX rewrites on a *copy* of device IR (contractions, transposes)
    codegen/
      module_builder.py  # module: phase funcs + memref-ABI entry func
      state.py           # LoweringState: value env, IV scope
      tensor_state.py    # SSA threading of host tensors through forall/for/if/while
      emit.py            # MLIR builder helpers (consts, casts, fill, empty, elementwise, reshape)
      registry.py        # @lowers(target) keyed by identity (Helion fn objects / OpOverloadPacket)
      ops/  memory.py slicing.py loops.py tiles.py creation.py contraction.py
            views.py casts.py masking.py
      aten/ helpers.py torch_mlir_import.py   # call-site typed helper cache + batch lowering
    support/ errors.py types.py einsum_spec.py
```

### 4.2 Key design elements

**A. `KernelGeometry` (authoritative, built once per config).** For each block id it records
the block size (from the config), extent (`BlockSizeInfo.size`), and `begin`/`end`/`step`
(from the loop node args). It also records the loop tree from
`ForLoopGraphInfo.block_ids` and each phase's grid ids. Nothing is inferred from tensor shapes.

**B. Loop lowering.**
- Each root graph becomes its own `scf.forall` over its own grid ids, and these are emitted in
  sequence. Sequential emission is a valid schedule for Helion's independent loops.
- The outer forall is **normalized**: `scf.forall (%t0, %t1) in (n0, n1)`, with
  `begin_b = lb_b + t_b * bs_b` computed via `affine.apply`.
  - `tile.begin`, `tile.end`, `tile.id` and `tile.index` all derive from `begin_b` and `t_b`,
    so they are unchanged.
  - Helion tiles within a phase are unordered and independent, so SFC reordering is legal.
- `_for_loop`/`_for_loop_step` become `scf.for` with `lb=begin`, `ub=end`, `step=bs|step`.
- `_if` becomes `scf.if`, and `_while_loop` becomes `scf.while`.

**C. Tensor-state threading (replaces deferred stores and the synthetic accumulator).**
- Pre-analysis (`tensor_effects`) finds, for every graph, the host tensors it reads and writes,
  including nested graphs.
- Within a phase function, `TensorState` maps each host tensor to its current SSA value.
- `scf.forall`:
  - `shared_outs` is the current value of every tensor the loop writes.
  - Each written tensor gets an **owned region**. It is taken from how the stores index the
    grid block ids: tile or point on grid dims, full extent elsewhere, and it must be
    consistent across all stores.
  - In the body, the tensor's state is `extract_slice(shared_out, owned)`.
  - The terminator emits one `parallel_insert_slice` per written tensor.
- Stores anywhere become `insert_slice` into the state, at offsets relative to the owned
  origin. Loads of a written tensor read the state, so read-after-write is correct.
- Nested `scf.for`, `scf.if` and `scf.while` carry or yield state only for the tensors they
  write.
- If disjoint ownership cannot be proven, fall back to a sequential `scf.for` over the grid
  that carries the full tensors. This is always correct, and a debug log records the fallback.
- This design removes `forall_insert_slices`, `ForStoreContext`, the synthetic geometry,
  `_find_descendant_store`, positional stores, `_existing_output_value`,
  `OutputTensorResolver`, and the limitations on multi-output routing and incompatible
  geometry.

**D. Boundary tiles.** When `extent % bs != 0` for a dimension:
- Load: `extract_slice` with size `min(bs, extent - off)`, then `tensor.pad` back to `bs`
  (pad value 0, matching Helion's masked-load default).
- `_mask_to(x, v)`: a real `select(iota < valid, x, v)` as a `linalg.generic`.
- Store: `extract_slice` of the valid part, then `insert_slice`.

When the block size divides the extent, the IR matches today's (static fast path).

**E. Phases and the calling convention.** The whole kernel lives in one module:
- `@k__phase<i>(ins..., inouts...) -> (inouts...)` are private, pure tensor functions, one per
  phase. This keeps the GPU-ready launch boundaries.
- `@k` is the entry, emitted by the backend directly in memref-ABI form:
  - Arguments: `(inout memrefs..., in memrefs..., 0-d scalar memrefs...)`.
  - Inputs become `bufferization.to_tensor ... restrict`; inouts become
    `bufferization.to_tensor ... restrict writable`.
  - It calls the phases in order, threading SSA values; `hl.barrier()` is a no-op.
  - It commits each final inout value with
    `bufferization.materialize_in_destination ... restrict writable`.
  - It gets `llvm.emit_c_interface` via lighthouse's `Runner.make_function_callable`.
  - `BackendDriver(result_to_args=False)` is used.
- **Why not `result_to_args`:** it cannot express in/out tensors. It makes every input
  `restrict`, so passing one buffer as input and output violates its contract. It rejects
  returning an argument unchanged (a phase that leaves an inout untouched). It is tensor-only.
  The own entry is about 60 lines, uses the same ops `result_to_args` would emit, and passed
  both pipelines in the spikes. The Runner, `TorchMemoryManager`, pipelines and schedules all
  stay lighthouse's.
- Function args are every host tensor the device code references (declared params,
  host-computed tensors, closure or attribute expressions), plus runtime scalars passed as 0-d
  tensors (never baked-in literals).
- The Python driver always:
  1. runs the host prefix (Helion semantics: host code first),
  2. binds tensors by name or origin expression, making each contiguous (a no-op when it
     already is),
  3. calls `@k`,
  4. copies back non-contiguous inouts,
  5. returns the prefix's captured `return` value.

  This makes in-place updates, `out=`, pre-initialized outputs and any `return` expression
  work. It also removes the single-phase vs multi-phase split.

**F. Dispatch registry.** `@lowers(hl.load)`, `@lowers(aten.addmm)` and similar, matched by
identity (the `OpOverloadPacket` or the Helion API function object). A handler returns a value
or raises. A `NotApplicable` sentinel is allowed only for explicit pattern handlers. Every node
is lowered inside `with node.meta["location"]:`.

**G. Contraction canonicalization (on copied graphs).** One FX pass rewrites every
recognized contraction to an internal `helion_mlir::contract(spec, lhs, rhs, acc?)` node. One
emitter lowers it (details in Phase 2):
- Identity 2-D maps emit `linalg.matmul`, batched identity maps emit `linalg.batch_matmul`, and
  everything else emits `linalg.contract`. These are all lighthouse anchors.
- An accumulator present means DPS accumulation (`outs = acc`). This keeps the loop-carried
  value in place and the op a single bf16×bf16→f32 contraction for AMX.
- Without an accumulator, it zero-fills in the result dtype taken from `node.meta['val']`.
- Transposed operands fold into the indexing maps instead of emitting `linalg.transpose`.
- Patterns outside today's guards (alpha/beta ≠ 1, dtype or shape mismatch with the
  accumulator, broadcasting batch dims) are **not** fused. They keep the generic path, exactly as
  today.

**I. Backend-owned pipelines.** `pipelines/scalar.yaml` and `pipelines/opt.yaml` reuse
lighthouse includes and schedules verbatim. They deviate only where §9 lists it:
- one-shot bufferization with `allow-return-allocs-from-loops`,
- the deallocation pipeline in the scalar path.

Pipeline choice becomes a backend setting. The `HELION_MLIR_PIPELINE=1` env var stays as the
default source.

**H. ATen helpers v2.**
1. At the call site, turn the operand MLIR types into meta tensors and run `node.target` on
   meta to get the exact result type.
2. Build a helper key `(target, literal args, operand types)` and emit `func.call @helper_<hash>`.
3. After the function body is built, batch-lower all requested helpers in one torch-mlir run.
4. If the batch fails, bisect or lower each helper alone and report the culprit node with its
   source location.
5. Cache helpers per module and per process.

Runtime scalars (SymInt/SymFloat, tile positions) become tensor operands. Direct linalg
lowering is used only where torch-mlir blocks: index-typed scalar operands and casts (as
`linalg.generic`), with torch-mlir as the fallback. Helion IR is never mutated.

### 4.3 Capabilities to preserve (each gets a guard test in Phase 0)

| ID | Capability | Why it must stay special-cased |
|---|---|---|
| C1 | `torch.einsum` → one `linalg.contract` (including multi-reduction and rank-5 VNNI forms) | torch-mlir decomposes to transposes + reshapes + `batch_matmul` + add |
| C2 | `addmm`, `baddbmm`, `acc + mm/bmm/matmul`, `acc + einsum` → one contraction with `outs = acc` | torch-mlir emits a zero-init contraction + add; lighthouse needs an in-place yield and a single AMX-shaped op |
| C3 | Mixed precision in one op (bf16×bf16→f32 accumulator) | `is_amx_bf16_contraction` |
| C4 | Transposed contraction operands folded into maps | avoids a separate `linalg.transpose` in front of the anchor |
| C5 | Production blocked matmul kernels (`helion_mlir_cpu_utils`) produce `x86.amx.*` ops under forced AMX | AMX IR gate |
| C6 | f32 production kernels run correctly under the optimizing pipeline | end-to-end gate on this machine |
| C7 | Scalar-indexed dims dropped with an explicit `collapse_shape` (never size-inferred rank reduction) | avoids a native assertion when a kept dim has extent 1 |
| C8 | Elementwise with an index-typed scalar operand (`x + tile.begin`) lowered directly | torch-mlir rejects index scalars in `.Tensor` overloads |
| C9 | Static view/reshape and alias ops without a helper | no helper round-trip; keeps shapes static |
| C10 | Multi-phase `hl.barrier()` and host-tensor interop kernels | Helion semantics |
| C11 | Combined tiles, arbitrary-depth nested grids/tiles, packing kernels (`pack_a`/`pack_b`) | production kernels |
| C12 | `generate_mlir()` for IR inspection | public API |

---

## 5. Detailed phase plan

Each phase keeps the full suite green. Its regression probes flip from
`xfail(strict=True)` to passing, and its capability guards (§4.3) stay green. Issue IDs
(`I…`) refer to the traceability table in §6. Phases run in order.

- **Order:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10.
- **Dependencies:**
  - Phase 5 depends on Phase 2 (registry) and Phase 4 (runtime scalars).
  - Phase 6 depends on Phase 1 (geometry) and Phase 3 (stores through the tensor state).
  - Phase 9 depends on Phase 4 (runtime scalars), Phase 5 (call-site helpers) and Phase 6
    (dynamic slice sizes).
  - Phase 10's autotuning should cover dynamic kernels, so it follows Phase 9.

Every phase ends with the same checklist:
1. The full suite passes, with the changed isolated regression probes.
2. The golden IR diff is empty, or an intentional diff has been reviewed.
3. The AMX IR gate and the optimizing-pipeline suite pass.
4. The conformance sweep count is recorded.
5. Repo memory is updated.

### Phase 0: Safety net, gates and backend-owned pipelines

**Goal.** Freeze today's capabilities behind tests before changing any lowering, and take
ownership of the pipeline configuration later phases rely on.
**Addresses.** I22, I23, I24.

Work items:
1. `tests/conftest.py`:
   - `run_direct(kernel, *args)`: runs `@helion.kernel(backend="mlir")`.
   - `run_generated(kernel, args, config=None)`: runs `generate_mlir` + `execute_mlir`.
   - `check_kernel(kernel, ref_fn, args, *, configs=None, paths=("direct", "generated"),
     atol, rtol)`.
   - An `isolated` marker that runs the test body in a `spawn` subprocess, so a native crash
     is reported as a failure of that test instead of killing the session.
   - Replace the duplicated `_execute` helpers in `test_einsum_contract.py` and
     `test_reduce_ops.py`.
2. `tests/test_regressions.py`: every probe from §2 as an isolated
   `xfail(strict=True, reason="I<n>")` test. The list:
   - nonzero begin, `hl.grid` step,
   - ragged 1-D, ragged K, ragged row max, row softmax,
   - `acc + addmm(b, x, y)`,
   - store-then-load, nested + outer store,
   - `if t.begin == 0`, `torch.where`,
   - scalar arg called twice with different values,
   - partial write, in-place, `out=`, non-contiguous input,
   - `hl.dot`.

   Strict xfail forces each phase to flip its own tests.
3. Golden IR: `tests/golden/*.txt` and `tests/test_golden_ir.py`, run with `--update-golden`
   to regenerate.
   - Content per function: op names in order, result types, and contraction indexing maps.
     SSA names and locations are stripped.
   - Kernel set: elementwise, tiled `addmm` matmul, `acc + einsum` blocked matmul (plus bias and
     affine epilogues), `pack_a`, `pack_b`, multi-phase, combined tile, nested grid→grid→tile.
4. AMX IR gate: `tests/test_amx_ir_gate.py`, marked `slow`.
   - Lowers the bf16 `_matmul_blocked_kernel`, `_bias` and `_affine` kernels through the
     optimizing pipeline under
     `TargetInfo.override(features=host + ["amx_bf16", "amx_tile", "amx_int8"])`.
   - Applies stages until `x86_vectorization` and asserts `x86.amx.tile_mulf` is present.
   - Nothing is executed. This is already verified in `temp/review_probes/spike_amx_ir.py`.
5. Optimizing-pipeline execution suite: `tests/test_opt_pipeline.py`, isolated,
   `HELION_MLIR_PIPELINE=1`.
   - Covers f32 `helion_mlir_cpu_utils` `matmul` (plain, bias + relu, `trans_b`), `linear`,
     `bmm`, and a tiled elementwise kernel.
   - Uses tiles ≥ 32 only (I23).
6. Backend-owned pipelines in `_compiler/pipelines/` (they move under `runtime/` in Phase 4):
   - `opt.yaml`: today's `pipeline.yaml` with lighthouse's `x86_64/lower.yaml` inlined, so its
     bufferization stage can carry `allow-return-allocs-from-loops`.
   - `scalar.yaml`: bufferization with that option, lighthouse's `bufferization-cleanup.yaml`,
     `convert-linalg-to-loops`, and lighthouse's `llvm-lowering.yaml`.
   - The executor selects by name, with the env var as the default.
   - Already verified: 284/284 pass with this scalar variant, and the f32 production kernels
     pass with the opt variant. All other includes and schedules remain lighthouse's.
7. Record the small-tile reproducer (§9) in `docs/MLIR_LIMITATIONS.md`.
8. `scripts/conformance_sweep.py`: a curated table of CPU-feasible upstream examples as
   `(module, kernel, small-input factory, reference)`.
   - Start with add, softmax, matmul (`addmm` and `hl.dot`), rms/layer norm, sum, long sum,
     cross entropy, embedding, geglu, swiglu, jagged-free attention pieces.
   - Each runs in a subprocess and is classified pass, wrong, error or crash.
   - The baseline JSON is committed.
9. `support/debug.py` reads the four `HELION_MLIR_DUMP_*` env vars in one place.

- **Deletes:** none.
- **Exit:** the suite is green with the new guards, every xfail fails as expected, goldens and
  the AMX gate are green on unchanged lowering code, and the sweep baseline is recorded.
- **Risks:** isolation cost. Only crash-prone tests are isolated, and they run in parallel with
  `pytest -n`, which is optional.

### Phase 1: Read device IR correctly; normalized outer loops

**Goal.** Replace every block-id and bound heuristic with Helion's authoritative metadata,
and emit SFC-friendly outer loops.
**Addresses.** I1, I2, I3, I4.

Work items:
1. `analysis/geometry.py`:
   - `BlockGeometry(block_id, block_size, extent, kind)`.
     - `block_size` comes from `BlockSizeInfo.from_config(config)`.
     - `extent` comes from `BlockSizeInfo.size`, must be an int under static shapes, and
       otherwise raises `DynamicShapeError`.
   - `LoopGeometry(graph_id, block_ids, begins, ends, steps)`.
     - `block_ids` comes from `device_ir.graphs[gid].block_ids`.
     - `begins`/`ends` come from the `_for_loop` args.
     - `steps` comes from the `_for_loop_step` args or the block size.
   - `KernelGeometry.from_host_function(hf, config, env)` also records each phase's root graph
     ids and grid block ids.
2. `BuildContext`: replace `block_id_to_size`, `block_id_to_upper_bound` and
   `block_id_to_domain_size` with `ctx.geometry`. Temporary compatibility properties are removed
   by the end of the phase.
3. `lower_nested_for_loop`:
   - Block ids come from geometry.
   - `lb = begin`, `ub = end`, and step is the explicit step or the block size.
   - Register `_for_loop_step`.
   - Multi-dim loop nodes use their declared ids directly.
4. Normalized outer forall in `build_kernel_body`:
   - Trip count per grid dim is `ceil((end - begin) / bs)`.
   - Offsets are `begin + t * bs` via `affine.apply`.
   - `block_id_to_iv[bid]` holds the offset, so every consumer is unchanged.
   - `tile.id` uses `t` directly; grid dims (bs = 1) use `t` as the index.
5. `symbolic_shape_restoration`: stop reading `node.args[2]` as upper bounds; use geometry.
6. Replace silent sentinels with errors: `-1` for a dynamic block size, `0` for an unresolved
   `sym_size`, and the `for_block_id_stack` fallback (tile ops resolve via symbol origins, or
   raise).

- **Deletes:** `_find_reused_block_id`, `_resolve_multi_block_ids`,
  `_block_id_matching_extent`, the body-symbol scans in `lower_nested_for_loop`,
  `_resolve_block_upper_bounds`, the grid-id skip, `for_block_id_stack`. About -250 LOC.
- **New tests:**
  - `hl.tile(2, n)` in 1-D and 2-D.
  - Nested `hl.tile(b, e)`.
  - `hl.grid(0, n, 2)`.
  - `tile.begin`/`tile.id`/`tile.end` values stored to an output under the normalized forall.
  - SFC remap applied to a 2-D, 32-tile matmul under the opt pipeline (IR check).
  - The existing cross-phase stale-id tests.
- **Exit:**
  - I1–I4 probes pass.
  - Goldens are regenerated once; the reviewed diff is limited to the forall header and
    `affine.apply` offsets.
  - AMX gate and opt suite are green.
- **Risks:** tile-and-fuse behaviour with `affine.apply` offsets. `spike_sfc.py` verified it
  for a contraction. Keep an internal `normalize_forall` switch for this phase only.

### Phase 2: Registry, exact op matching, one contraction path (including `hl.dot`), diagnostics

**Goal.** One identity-keyed dispatcher. One contraction lowering that preserves C1–C4 and
adds `hl.dot`. Errors that point at the kernel source.
**Addresses.** I5, I6, I7, I8, I9 (dispatch part), I10 (casts), I25.

Work items:
1. `codegen/registry.py`:
   - `@lowers(*targets, overloads=None)`, keyed by object identity.
     - Helion API functions: `memory_ops.load/store`, `_tracing_ops.*`, `tile_ops.*`,
       `creation_ops.full`, `matmul_ops.dot`, `view_ops.subscript`.
     - ATen: `torch._ops.OpOverloadPacket` with an optional overload filter, so `div.Tensor`
       does not match `div.Tensor_mode`.
   - One `lower_node(ctx, node)`: `with node.meta["location"]` → registry → generic ATen helper
     → `UnsupportedOperationError`.
2. Move handlers into `codegen/ops/{memory,loops,tiles,creation,views,casts,contraction}.py`.
   This is mechanical.
3. `analysis/canonicalize.py` works on graph copies (`GraphInfo.copy`; `meta` keys such as
   `location` and `tile_with_offset` are kept). It rewrites these patterns to
   `helion_mlir::contract(spec, lhs, rhs, acc?)`:

   | Pattern | Condition (anything else is left untouched) |
   |---|---|
   | `mm`/`bmm`/`matmul(a, b)`, 2-D/3-D, same batch | — |
   | `addmm(acc, a, b)`, `baddbmm(acc, a, b)` | beta = alpha = 1; acc shape/dtype = result |
   | `add.Tensor(acc, X)` or `(X, acc)`, alpha = 1, X ∈ {mm, bmm, matmul, einsum, contract without acc} | X single-use; acc shape = result; valid accumulator dtype |
   | `hl.dot(a, b, acc=, out_dtype=)` | acc given → fused; otherwise result dtype from `meta['val']` |
   | `helion_mlir::einsum(eq, [a, b])` | contractible per `einsum_spec` (as today) |
   | Operand is `permute`/`transpose`/`t` of a tile | single-use; folded into the operand map |

4. `codegen/ops/contraction.py`: one emitter from the spec.
   - Identity maps emit `linalg.matmul` or `batch_matmul`; everything else emits
     `linalg.contract`.
   - With `acc`, the op writes into `outs = acc` (C2, C3).
   - Without `acc`, it zero-fills f32 for float inputs and then applies `truncf` to the result
     dtype. This fixes bf16 accumulation and keeps the contraction AMX-shaped.
5. `codegen/emit.py`:
   - Shared constant, zero-attribute, empty, fill and reshape builders, plus the scalar-dim
     `collapse_shape` (C7).
   - `cast_tensor` becomes an all-parallel `linalg.generic` with `arith` casts, replacing
     `tensor.generate`.
6. Diagnostics:
   - Backend errors derive from `helion.exc.BaseError` and are raised inside
     `node.meta["location"]`, so Helion prints the kernel's file and line.
   - Remove the misleading "static_shapes" hint.
   - Replace the 24 broad `except Exception` blocks with explicit checks (e.g.
     `isinstance(v.type, ir.RankedTensorType)`) or narrow exception types.
7. Imports and cleanup:
   - Use module-level `mlir` imports in codegen and analysis modules; the guarded import stays
     in `inject.install()`.
   - Delete dead code.
   - `uint8` raises a clear unsupported error instead of emitting `ui8`.
8. `mul(x, None)`: instrument which kernels reach it, then fix the real cause (likely an
   unresolved scalar literal) or delete it if nothing reaches it.

- **Deletes:**
  - `node_dispatch.py` and the ~20 `codegen.py` wrapper methods.
  - `lower_custom_aten`, `aten_target_matches`, the string lists in `transpose_ops.py`.
  - `resolve_contraction_operand`, `lower_add_matmul_accumulate`, `lower_addmm`,
    `lower_baddbmm`.
  - `matmul_ops.py`/`einsum_ops.py`, merged into `ops/contraction.py`.
  - About -600 LOC.
- **New tests:**
  - `acc + addmm(b, x, y)`.
  - `hl.dot` with and without acc, bf16 → f32 acc, `out_dtype`, transposed rhs, and 3-D
    batched.
  - IR assertions: exactly one contraction op, expected maps, and `outs` is the loop iter arg.
  - `x // 3` on float tensors.
  - An error message that contains the kernel source line.
  - Registry overload-filter unit tests.
- **Exit:**
  - I5–I8, I10 (casts) and I25 probes pass.
  - Goldens are unchanged except the reviewed cast lowering.
  - AMX gate green (C1–C5).
- **Risks:** graph copies must be the only graphs the backend reads. Helion's own IR stays
  untouched, which a test asserts.
- **Outcome (done).** I5, I6 and I25 probes pass; I7, I8 and I10 (casts, `tile.index`)
  are covered by `tests/test_contractions.py` and `tests/test_registry.py`. Goldens,
  AMX gate and opt suite unchanged. Deviations from the items above:
  - Contractions are matched into a read-only side table (`analysis/contractions.py`)
    instead of rewriting graph copies. Absorbed nodes (`mm` under `acc + mm`, folded
    transposes) are simply not lowered, so Helion's IR is never touched and no copy is
    needed.
  - Handlers stay in the existing `lowering/` package (`registry.py`,
    `contraction_ops.py`, `emit.py`, ...) rather than moving to `codegen/ops/`.
  - `mul(x, None)` is Helion's `strip_unused_inputs`, which drops repeated inputs.
    `_normalize_aten_args` now restores them for any pointwise op with one distinct
    input (not only `mul`).
  - Broad `except Exception` blocks remain only in `execution.py` (Phase 4, item 5) and
    the ATen bridge modules that Phase 5 deletes.

### Phase 3: Tensor-state threading

**Goal.** Stores and loads follow SSA through every loop level. This replaces deferred
terminal inserts and the synthetic accumulator.
**Addresses.** I11, I12 (outputs), I9 (state split).

Work items:
1. `analysis/tensor_effects.py`:
   - `HostTensorRef` gives a canonical identity per host tensor: the FakeTensor identity plus
     its origin.
   - Per graph, it records the refs read and written, transitively through `_for_loop`, `_if`
     and `_while_loop` bodies.
   - For every store, it records the index descriptors from `resolve_index_descriptor`.
2. Ownership, per root graph and written ref:
   - A dim is `grid(bid)` if every store to the ref indexes it with the same grid block id
     (tile or point). Otherwise it is `full`.
   - Parallel emission requires every forall grid id to appear in every written ref's owned
     dims. Otherwise the root graph falls back to a sequential `scf.for` nest carrying full
     tensors, which is always correct and logged at debug level.
3. `codegen/tensor_state.py` provides:
   - `current(ref)` and `origin(ref)` (per-dim offset of the local tile),
   - `load(ref, plan)` and `store(ref, plan, value)`, with plans rebased to the origin,
   - `scope()`, which reports the refs written inside a region,
   - `carry(refs)` and `rebind(refs, values)`.
4. One normalized forall per root graph, emitted in sequence:
   - `shared_outs` are the current states of the written refs.
   - Body entry takes `extract_slice(shared_out, owned)`.
   - The terminator emits one `parallel_insert_slice` per written ref.
   - Different grids in one phase are now legal.
5. `scf.for` iter args are Helion's carried values plus the states of refs written in the body.
   Contraction accumulators stay DPS (C2). Other carried updates are legal thanks to Phase 0's
   bufferization option.
6. A load of a ref written in an active scope reads the state (read-after-write). A load of a
   read-only ref reads the function argument, as today.
7. A store is an `insert_slice` into the state with a plan relative to the origin, cast via
   `emit.cast_tensor`.
8. `BuildContext` splits into `KernelGeometry` (Phase 1), `LoweringState` (value env, IV scope)
   and `TensorState`.
9. Migration: the new path is gated by an internal module constant. Both paths run in CI during
   the phase. Flip the default, then delete the old path and the constant before the phase
   closes.

- **Deletes:**
  - `forall_insert_slices`, `for_store_ctx_stack`, `for_store_context.py`.
  - `_prepare_synthetic_accumulator`, `_compute_synthetic_tile_geometry`,
    `_find_descendant_store`, `_flush_synthetic_accumulator_to_parent`,
    `_block_id_to_out_dim_from_terminal_store`.
  - `_existing_output_value`, `_validate_parallel_insert_fits`, `_validate_multi_output_shapes`.
  - The positional and bound store paths in `memory_ops.py`.
  - `output_resolver.py`: outputs are simply the written refs.
  - About -700 LOC.
- **New tests:**
  - Store-then-load; two stores to one region (last wins); nested + outer stores.
  - Multi-output with a nested reduction.
  - Independent top-level loops with different grids in one phase.
  - A partial-tile write preserving `shared_out` content.
  - Ownership fallback (a write not indexed by a grid dim) giving the correct sequential result.
  - Goldens: full-tile-store kernels canonicalize to today's IR (0 `insert_slice` left, as in
    the spike).
- **Exit:** I11 probes pass, goldens are equal after canonicalization, AMX gate and opt suite
  are green, and the packing kernels (C11) are correct under both pipelines.
- **Risks:** tile-and-fuse on partially written owned tiles, covered by the packing kernels in
  the opt suite. Rollback is the internal constant during the phase.
- **Outcome (done).** Both I11 probes pass; tests in `tests/test_tensor_state.py`, and the
  former "incompatible geometry" diagnostic test now checks two sequential foralls
  numerically. 9 of 10 goldens are unchanged after canonicalization; `nested_grid` now
  carries its owned `[1, 3, 32]` region through the nested loops instead of zero-filled
  synthetic accumulators. AMX gate and opt suite green. Deviations:
  - No dual-path migration constant: the old path was replaced in one step with the suite
    green before and after (git is the rollback).
  - `BuildContext` is still one object; the analyses (`geometry`, `effects`,
    `contractions`) and the new `TensorState` are separate attributes rather than a split
    `LoweringState`.
  - Function signature for this phase: inputs are the tensor params that are not written
    or are also loaded; outputs are every written host tensor in first-store order.
  - `plan_slice` gained static partial slices (`out[t, :n1]`), which the old positional
    store path handled implicitly. Scalar stores (`out[t] = 0.0`) now fill the slice.
  - Gap made explicit, not new: a stored value with size-1 dims broadcast against the slice
    (`out[tm, tn] = x[tm, None]`) is legal Helion (type propagation checks rank only; eager
    mode and Triton `tl.store` broadcast). It failed in the MLIR pipeline before Phase 3
    and now raises the "transposed or mismatched tile layout" error. Scheduled in Phase 7
    (I27); probe: `temp/probe_store_broadcast.py`.

### Phase 4: Host semantics, calling convention, runtime

**Goal.** Helion semantics on every call path. In/out semantics explicit in the MLIR. One
module per compiled config.
**Addresses.** I12 (rest), I13, I14 (ABI), I15, I16.

Work items:
1. `analysis/signature.py`, `KernelSignature.from_host_function(hf, effects)`:
   - Ordered refs: declared tensor params in declaration order, then other host tensors in
     first-use order.
   - Role is `inout` if the ref is written anywhere, otherwise `in`.
   - Runtime scalars are non-constexpr int/float params used in device code. They become 0-d
     tensor args (i64/f32/f64).
   - Records the in/inout subsets per phase.
2. Module builder:
   - `@k__phase<i>` are private tensor functions (phase ins + inouts → phase inouts).
   - `@k` is the memref-ABI entry described in §4.2 E, lowered with
     `BackendDriver(result_to_args=False)`.
   - Remove `_extract_result_metadata_pre_lowering`.
3. `driver.py` replaces the two paths in `bound_kernel.py`. At each call:
   1. Run the host prefix.
   2. Resolve each ref: a name in the prefix locals, otherwise evaluate
      `origin.host_str()` in the prefix namespace (closure, attribute and item origins).
   3. Wrap scalars as `torch.tensor(v, dtype)`.
   4. Make tensors contiguous with `t if t.is_contiguous() else t.contiguous()`.
   5. Guard aliasing: refs whose storage overlaps get a cloned input, because `restrict` must
      hold; overlapping inouts raise a clear error.
   6. Call the compiled entry.
   7. Copy back into non-contiguous inouts.
   8. Return the prefix's captured `return` value, whatever expression it is.
4. `api.generate_mlir`:
   - Copies `Settings` instead of mutating the kernel's.
   - Returns the full module (phases + entry).
   - `execute_mlir` keeps working for kernels that only use declared params.
   - New `compile_mlir(kernel, args, config=None) -> Callable` shares the driver.
5. Executor:
   - Stateless `compile(module, entry)`.
   - Pipeline chosen by name.
   - Errors keep their type and get context notes (`add_note`) instead of a blanket
     `RuntimeError`.
6. Document the ABI in `docs/MLIR_DESIGN.md`:
   - argument order and `restrict`/`writable` meaning,
   - why not `result_to_args`,
   - the contiguity trade-off: free when contiguous; otherwise one copy in, plus a copy back
     for inouts. Strided memrefs are deferred because they are untested in the lighthouse
     pipelines.

- **Deletes:**
  - `_build_multi_phase_driver` name threading, `requires_multi_phase_driver`.
  - `build_phase_modules`/`PhaseModuleResult` and the per-module JIT.
  - `_extract_return_names`, executor output allocation.
  - The parts of `phase_plan.py` superseded by the signature.
  - The single vs multi dispatch in `mlir_compile_config`.
  - About -400 LOC.
- **New tests:**
  - Partial write into `full_like(7)`.
  - Carried over from Phase 1: drop the `region=` comparisons in
    `tests/test_loop_geometry.py` and compare full outputs. They exist only because the
    `zeros_like` initialization is lost today.
  - In-place `x[t] += 1`, both returning `x` and returning `None`.
  - `out=` parameter.
  - Float `alpha` called with 2.0 then 3.0 on the same bound kernel; the same with an int.
  - Return expressions: `.view(...)`, a tuple, `None`.
  - Non-contiguous input and non-contiguous `out=`.
  - Aliased args `k(x, x)`.
  - A closure tensor loaded through a subscript (or a documented Helion restriction).
  - `generate_mlir` leaves `kernel.settings` unchanged.
  - The multi-phase suite (C10).
  - Carried over from Phase 3:
    - A written param that is not loaded still starts from `tensor.empty`; with inouts it
      starts from the argument (partial writes into `out=`).
    - Multi-phase `build_phase_plans` still drops a phase's writes to *declared* params
      from its outputs; with inouts they are threaded like any other tensor.
    - Two host names for one storage (a host-side view of a written tensor) are separate
      states; resolve refs by origin/storage so they share one.
    - Return the kernel's `return` expression instead of "every written tensor".
- **Exit:** I13–I16 probes pass and there is one module per config.
- **Risks:** the host prefix runs on every call. This matches Helion's Triton wrapper. Its
  limitations (a single trailing `return`) stay documented.
- **Outcome (done).** I13, I14 (ABI) and I15 probes pass (partial write into `full_like(7)`,
  in-place with and without `return`, `out=`, float and int scalars changing between calls,
  strided input and strided `out=`); tests in `tests/test_calling_convention.py`,
  `tests/test_signature.py`, `tests/test_host_code.py`. One module per config;
  `bound_kernel.py`, `phase_plan.py`, per-phase modules and executor output allocation are
  gone (backend net -310 LOC). All 10 goldens changed only at the function boundary
  (memref signature, `to_tensor` instead of `tensor.empty`, `materialize_in_destination`).
  AMX gate and opt suite green. Sweep 8 -> 11 of 22 (`broadcast_matmul`, `geglu`, `swiglu`
  return a view expression). Deviations and findings:
  - The host prefix became a generator (`host_code.py`): it yields its locals where the
    device loops were and resumes after the entry call, so statements after the loops and
    the `return` expression run after the kernel, and early returns work. It copies
    Helion's AST (no mutation, so every config can rebuild it), replaces host block sizes
    (`hl.register_block_size`, a `BlockSizeType`) by config values, and resolves globals
    through `hf.global_imports` (`_source_module.W`). Closures are a Helion restriction.
  - Entry arguments carry `{helion.name, helion.role, helion.param}`; the driver evaluates
    each `helion.name` (the host expression) in the host locals, and `execute_mlir` maps
    `helion.param` to its tensor arguments. `execute_mlir` zero-initializes host-created
    inouts, rejects host-computed inputs and scalars, writes declared parameters in place
    and returns every inout.
  - Read-only host views of a declared parameter stay MLIR reshapes (aliases); every other
    host tensor is its own argument. Inouts that share memory are rejected at the call.
  - Runtime scalars are `f64`/`i64` 0-d arguments consumed by `add`/`sub`/`mul`/`div`;
    nodes with a `SymFloat` operand are no longer sent to torch-mlir helpers (Phase 5 for
    other ops).
  - The executor stayed in `_compiler/execution.py` (`compile_entry`, `entry_args`); the
    pipeline is chosen by name (`"scalar"`/`"opt"`) with the env var as default (Phase 10
    makes it a config key).
  - Found and fixed: a `tile.id`/`tile.end`/`tile.count` load/store index used the tile
    offset (`IndexDescriptor.kind`; only `grid`/`tile.begin` equal the offset, so only they
    can own a forall dimension).
  - Found, not fixed (Phase 7): a host tensor whose shape depends on a block size
    (`torch.zeros((m, n // block_n))`) gets a dynamic MLIR type and fails to lower.

### Phase 5: ATen helpers v2

**Goal.** Helper signatures come from real operand types. No shape guessing, no mutation of
Helion IR, and failures isolated per node.
**Addresses.** I14 (helper side), I17, I10 (gathers).

Work items:
1. `codegen/aten/helpers.py`:
   - `HelperRequest(target, literal_args, operand_types)`.
   - At the call site, operand MLIR types become meta tensors, and running `node.target` on
     them gives the exact result types.
   - Emit a private `func.func` declaration and a `func.call`.
2. Runtime scalar operands (SymInt/SymFloat nodes, tile positions):
   - They are materialized as 0-d tensors (`tensor.from_elements`) and passed as helper
     operands; their FX placeholders get 0-d values.
   - Index-typed scalar binary ops keep their direct lowering (C8).
   - Since Phase 4, runtime scalar parameters are entry arguments (`ctx.scalars`, `f64`/
     `i64`) and `BuildContext.has_symbolic_operand` keeps nodes with a SymInt/SymFloat
     operand out of the helper batch; only `add`/`sub`/`mul`/`div` lower them today.
     Scalar precision: torch computes `bf16_tensor * float` in f32 (opmath), the direct
     lowering truncates the scalar to the tensor dtype first; keep that in mind for the
     helper path.
3. After all functions are built, one torch-mlir batch covers the unique requests. On failure,
   lower each request alone to find the culprit. Report it with the node's source location,
   op and operand types.
4. A process-wide helper cache keyed by the request is parsed once into the shared context and
   cloned per module.
5. An explicit whitelist of direct lowerings, each with a one-line justification in code:
   - contractions (C1–C4),
   - transpose/permute,
   - static view/reshape/alias (C9),
   - casts,
   - index-scalar binary ops (C8),
   - einsum capture.

   Everything else goes through helpers.
6. Gathers (`lower_flat_gather`, subscript gather) go through the torch-mlir `index.Tensor`
   helper first. A direct `linalg.generic` is used only if torch-mlir blocks.
   Carried over from Phase 3: gathers and subscripts of a host tensor read the function
   argument, not its tensor state, so a gather after a store to the same tensor misses the
   store. Route them through `TensorState` like `load`.
7. A test asserts Helion `node.meta` is identical before and after compile.

- **Deletes:**
  - The shape reconstruction in `aten_lowering.py` (about 700 LOC).
  - `aten_prepass.py`, `helper_rebuild.py`, the signature matching in `AtenHelperTable`,
    `_prebuild_aten_helpers`.
  - `symbolic_shape_restoration.py`, after an instrumented zero-hit audit.
- **New tests:**
  - Row softmax, `torch.where`, amax/sum/argmax/mean, rank-broadcast add.
  - Carried over from Phase 1: `gather_gemv` in the conformance sweep. Its `tile_n_s.index`
    inside the inner loop is a `sym_size` of a loop-body placeholder that is not an iter arg.
    That node reaches lowering concretized and untagged, so its block id is lost (it used to
    fall back silently to the innermost loop's block). Removing
    `symbolic_shape_restoration` and meta mutation should fix it; confirm with a test.
  - Carried over from Phase 2: a repeated operand in a non-pointwise or multi-input op
    (`x @ x` arrives as `mm(x, None)`, `where(c, x, x)` as `where(c, x, None)`) is not
    recovered. Take the operand identity from the helper request instead of the stripped
    args, and let `analysis/contractions.py` see the restored operands.
  - A SymFloat scalar in elementwise code; gelu/sigmoid/exp/erf.
  - One unsupported op produces a precise error without affecting others.
  - Metadata unchanged; exactly one torch-mlir run per compile; a helper cache hit across
    configs.
- **Exit:** I17 probes pass, `aten_lowering.py` is at most about 250 LOC, and suite compile time
  is no worse (measured).
- **Risks:** exotic ops without meta kernels fall back to `FakeTensorMode` evaluation.
- **Outcome (done).** I17 probes (`row_softmax`, `where_relu`) pass; new tests in
  `tests/test_aten_helpers.py`. `aten_lowering.py`, `aten_prepass.py`,
  `symbolic_shape_restoration.py`, the four old `aten_bridge` modules and `load_ops.py` are
  gone; `aten_bridge/helpers.py` (about 450 LOC) replaces them (backend 6770 -> 5394 LOC).
  Suite 40-48 s parallel, the same range as before. Goldens unchanged; AMX gate and opt suite
  green. Sweep 11 -> 20 of 22 (`cross_entropy`, `embedding`, `gather_gemv`, `layer_norm`,
  `rms_norm`, the three softmaxes, `welford`). Deviations and findings:
  - Helion's metadata is read, never modified, so the zero-hit audit of
    `symbolic_shape_restoration` was moot: nothing reads concretized metadata any more.
    `sym_size.int` takes the operand's MLIR dimension, `view`/`reshape` use `infer_results`
    (meta evaluation on the operand types), `hl.zeros` shapes take tile extents or constant
    values and otherwise raise (no silent size-1 fallback).
  - Runtime scalars are passed to helpers as `f64`/`i64`/`i1` scalars (torch-mlir imports
    them as `!torch.float`/`!torch.int`/`!torch.bool`), not as 0-d tensors;
    `has_symbolic_operand` is gone. The direct index-scalar binary lowering now defers to the
    helper when the result dtype differs from the tensor's (type promotion).
  - Stripped inputs are restored by recording each node's arguments just before Helion's
    `strip_unused_inputs` (monkeypatch scoped to MLIR environments) and filling only the
    `None` positions, so later rewrites (`_mask_to`) are kept. `analysis/contractions.py`
    matches on the restored arguments (`x @ x`).
  - A node's helper is `_aten_<op>_<hash>` of target, literals and operand types; the
    lowered function type is checked against the call site. The per-request fallback also
    fills the cache, so one bad op does not block the others.
  - Gathers: a load with one index tensor becomes an extract slice (whole gathered
    dimension) plus the `aten.index.Tensor` helper, from the tensor's current state. More
    than one index tensor, or a non-1-D one into an N-D tensor, is rejected (Helion and torch
    disagree on the result shape there), as are tensor-indexed stores. torch-mlir fails on
    `i32` index tensors, so indices are widened to `i64`. The `trailing_extent` clamp is gone.
  - Subscripts of device values are one extract slice plus a reshape; `x[None, :]` was
    mis-indexed before.
  - Found and fixed: Helion maps `aten.gelu` to its own `_gelu_erf`/`_gelu_tanh_approx` ops
    (lowered as the `aten.gelu` helper); `_inductor_lowering_extra` nodes (Inductor
    intermediates such as the sum inside `mean`) have no value, since the op's helper
    recomputes them.
  - Found, not fixed: `matmul_layernorm` calls `hl.specialize` in host code (I29, Phase 7);
    `concat2d_dim1` needs ragged tiles and `extra_mask` (Phase 6).

### Phase 6: Boundary (ragged) tiles

**Goal.** Correct results for non-divisible extents with static tile types. IR is unchanged
when the block size divides the extent.
**Addresses.** I18.

Work items:
1. Geometry marks `ragged(bid)` when `(end - begin) % bs != 0`. The per-iteration valid size is
   `v = affine.min(bs, end - offset)`, emitted only for ragged dims.
2. `SlicePlan`: `DimSlice.size` becomes `int | ir.Value`. A ragged load is a dynamic
   `extract_slice` followed by `tensor.pad` (high = `bs - v`, value 0) back to the static tile
   type.
3. `_mask_to(x, other)`:
   - Each dim of `x` maps to its block id through the symbol origin of its SymInt.
   - Ragged dims get a `linalg.generic` with `linalg.index`, `cmpi ult` and `select`.
   - It is the identity when there are no ragged dims.
4. A ragged store takes an `extract_slice` of the valid sub-tile, then an
   `insert_slice`/`parallel_insert_slice` with dynamic sizes.
5. `extra_mask`:
   - On load: `select(mask, loaded, 0)`.
   - On store: `select(mask, value, current slice)`, then insert.
6. `tile.index` sizes, the `tile.end` clamp and `tile.count` all come from geometry.
   Found in Phase 5: `_get_symnode(block_size_N)` lowers to the block size as a value but
   to the tile extent in `hl.zeros` shapes; make them agree (e.g. `x.view(tile_m, -1)`).
7. Remove the "ragged combined-tile" rejection and any `_validate_*` check this makes redundant.
   Update finding 2 in `docs/PADDING_FUSION_FINDINGS.md`.

- **New tests:**
  - Ragged 1-D (100/16), ragged combined 2-D, ragged-K matmul.
  - Ragged row max/sum/softmax, which exercises `_mask_to` identity values.
  - Nonzero begin combined with ragged, `extra_mask` on load and store.
  - Sweep: `concat2d_dim1` (found in Phase 5).
  - The padding doc's `_read_oob_1d_test` must return zeros.
  - Non-divisible property fuzzing, isolated.
  - Ragged kernels under the opt pipeline with tiles ≥ 32.
- **Exit:** I18 probes pass, the fuzzer finds no crashes, and static-shape goldens are
  unchanged.
- **Outcome (done).** The three I18 probes pass; tests in `tests/test_ragged_tiles.py`; the
  combined-tile matmul fuzzer samples non-divisible shapes and runs isolated. Goldens
  unchanged, AMX gate and opt suite green, a ragged M/N/K matmul and 2-D elementwise pass
  under the opt pipeline. Sweep 20 -> 21 of 22 (`concat2d_dim1`). Deviations and findings:
  - A tile keeps `tile_extent = min(block size, span)`, not a full block: padding a short
    loop to its block wastes work (the CPU utilities use blocks of 1024/4096 over shorter
    loops on purpose). The block-size symbol now lowers to the tile extent too, so shapes
    and values agree (item 6). Only a kernel doing arithmetic with the block size of a
    loop shorter than its block sees a different value than under Triton.
  - Raggedness is decided per loop from its bounds (`BuildContext.bind_loop`): static
    divisible bounds give no valid size; a runtime bound always gets
    `affine.min(tile, end - offset)`. `geometry.is_ragged` is gone.
  - `tile_window` also clips a tile to the tensor when the loop may run past it (a
    tensor smaller than the iteration domain, e.g. `concat2d_dim1` and the padding doc's
    `_read_oob_1d_test`); the excess reads as zeros. A negative offset that may point
    before the tensor start is rejected.
  - A forall owns only the real part of a partial tile, so the thread-local state of a
    written tensor has dynamic dims there; loads and stores inside read their size with
    `tensor.dim`.
  - `extra_mask` goes through the `aten.where` helper (on store, against the current
    slice).
  - A tile index of a block id without an active loop, silently given a static size
    before, now raises (never hit in the suite, sweep or examples).

### Phase 7: Coverage

**Goal.** Close the remaining Helion language gaps, ordered by the conformance sweep.
**Addresses.** I19, I20, I21, I27, I28, I29.

Work items:
1. `_if` → `scf.if`:
   - Scalar predicates come from index comparisons (`eq`, `ne`, `lt`, and so on as
     `arith.cmpi`) and `_and`/`_or`/`_not` (`arith`).
   - Tensor predicates (`predicate_is_tensor`) raise a clear unsupported error.
   - Branches are lowered inside tensor-state scopes. Yields follow
     `IfGraphInfo.branches_outputs` plus the written states, and `_phi` merges them.
2. `_while_loop` → `scf.while`: the condition graph fills the before-region and the body the
   after-region. Carried values are Helion's plus the written states.
3. `_constant_tensor` → a splat constant or fill.
4. `hl.reduce(fn, x, dim)` → `linalg.reduce`, with the combiner body lowered from the combine
   FX graph. `hl.associative_scan`/cumsum go through a torch-mlir helper for known combiners,
   otherwise a sequential `scf.for` scan.
5. `hl.split`/`hl.join` → slice, insert and reshape.
6. Broadcasting stores (I27): `out[tm, tn] = x[tm, None]` is legal Helion (rank must match,
   sizes may be 1). `memory_ops._store_value` emits one `linalg.generic` whose input map
   uses constant 0 for the size-1 dims and whose `outs` is the destination slice, so the
   broadcast is never materialized; the dtype cast folds into the same body. Rank mismatch
   and non-1 size mismatch (transposed layouts) keep the current error.
7. Atomics, `hl.rand`, `inline_asm` and `device_print` raise explicit unsupported errors with the
   reason. Atomics need memref-level semantics and are out of scope for tensor IR.
8. Block-size-dependent host shapes (I28, found in Phase 4): a host tensor such as
   `torch.zeros((m, n // block_n))` with `block_n = hl.register_block_size(n)` has a
   symbolic dim in its fake. Specialize ref types with the config's block sizes (the value
   the host code computes at run time) instead of emitting a dynamic `?` dim.
9. Host-side Helion API calls (I29, found in Phase 5): `host_code.py` copies
   `hl.specialize(y.size(1))` verbatim, which raises `NotInsideKernel` at run time. Replace
   the call by its specialized value, as Helion's host codegen does (`repr` of the
   specialized proxy); other host API calls get an explicit error.

**Carried over from Phase 5:** `concat2d_dim1` (sweep) needs Phase 6 (a load of a
16-wide tensor with a 32-wide tile, `extra_mask`). Fixed in Phase 6.

- **Exit:** every curated sweep example that does not use atomics or rand passes. The concrete
  target is set from the Phase 0 baseline.
- **Outcome (done).** Sweep 21 -> 22 of 22 (`matmul_layernorm`); the last known-gap probe
  (`tile_if`) passes, so the regression file has no xfail left. New tests in
  `tests/test_control_flow.py` and `tests/test_language_ops.py`, probes `broadcast_store` and
  `block_size_host_shape`, and a host-code test for `hl.specialize`/`hl.register_tunable`.
  Deviations and findings:
  - `_if` lowers each branch into a draft `scf.if` first, since the result types are only
    known after lowering, then moves the ops into the real one. `_if`'s outputs (then-side,
    then else-side values) both map to the same `scf.if` results; `_phi` picks them. A
    one-element tensor condition is read; larger tensor conditions are rejected.
  - Scalar arithmetic and comparisons (`operator.*` on `SymInt`s, `_and`/`_or`/`_not`) were
    not lowered at all before; `lowering/scalar_ops.py` covers them with Python floor
    division and modulo semantics. Helion itself evaluates `not tile.begin == 8` statically
    through a guard, so such a condition never reaches the backend as an `_if`.
  - `_while_loop` → `scf.while`. Helion unrolls counted loops, so only data-dependent loops
    reach it; `while`/`else` is rejected by Helion.
  - `hl.reduce`/`hl.associative_scan` (and `torch.cumsum`, which Helion maps to the latter)
    use one sequential `scf.for` for every combiner instead of torch-mlir helpers for known
    ones: one path, and scans are rarely hot. Tuple inputs are rejected.
  - Broadcasting stores (I27) write one `linalg.generic` into the destination slice;
    loads now also honor `None` (new axes) in their index, which `x[tm, None]` needs.
  - Host tensor types (I28) substitute the config's block sizes into the fake's symbolic
    sizes; a size depending on anything else raises a `DynamicShapeError`.
  - Host code (I29) replaces `hl.specialize` and `hl.register_tunable` by their values
    (the latter needs the config, now passed to `build_host_function`); any other Helion
    API call in host code is rejected.
  - Atomics, `hl.rand`/`rand4x`/`randint`, `inline_asm_elementwise`, `inline_triton` and
    `device_print` raise with their reason (`lowering/unsupported_ops.py`).

### Phase 8: Performance

**Goal.** Measurable speedups on the existing kernels, without changing what compiles.
**Addresses.** I33 (and part of I26's performance side).

Work items:
1. Fewer per-iteration allocations. Carried values that are not contractions (running
   `max`/`sum`, softmax statistics, `acc = acc + x` through a helper) are new tensors
   every iteration; `allow-return-allocs-from-loops` accepts that, at one allocation and
   copy per iteration. Contractions already update their iter arg in place (DPS).
   - First measure: count `memref.alloc` inside loops in the bufferized IR and time the
     affected sweep kernels (`layer_norm`, `rms_norm`, `softmax_two_pass`, `welford`,
     `longsum_w_red_loop`, `cross_entropy`) and the ragged row max, at sizes where a loop
     runs many iterations.
   - Make the loop update destination-passing: when the result of a node is the value
     yielded for an iter arg and has the iter arg's type, lower it with that iter arg as
     `outs` (elementwise and reduction ops directly as `linalg.generic`/`linalg.reduce`
     with `outs = iter arg`; a torch-mlir helper gets an extra `outs` operand only if
     one-shot bufferization cannot already make it in place after inlining — check
     first, the pass may do it once the helper is inlined).
   - Exit: no allocation left inside the measured loops, or a documented reason per
     remaining one; timings recorded before/after.
2. `hl.reduce` and `hl.associative_scan` with known combiners. Both are one sequential
   `scf.for` per step today (Phase 7).
   - A combine graph that is a single `add`/`mul`/`maximum`/`minimum`/`logical_and`/
     `logical_or` of its two placeholders (matched structurally, like contractions)
     becomes a `linalg.reduce` with the matching `arith` body and the op's identity as
     init; other combiners keep the loop.
   - Scans: check whether torch-mlir lowers `aten.cumsum`/`cummax` to linalg-on-tensors
     (it may emit `tm_tensor.scan`, which lighthouse cannot lower); if not, keep the
     sequential loop but carry the whole row per step so the non-scan dims vectorize,
     and measure.
   - Exit: same results as today (tests in `tests/test_language_ops.py`); timings before
     and after for a reduction and a cumsum over a long dim.
- **Outcome (done).** Suite 408 -> 423, sweep 22/22, examples pass. Timings on this laptop
  varied several-fold between identical runs, so the exit criterion is the allocation
  count instead (agreed with the user). `memref.alloc` ops inside loops
  (`temp/phase8/measure_allocs.py` and `opt_allocs.py`, 256-row inputs), before -> after:

  | Case | Scalar bufferization | Opt pipeline |
  |---|---|---|
  | `row_sum_loop` (`acc + x.sum(-1)`) | 3 -> 0 | 2 -> 0 |
  | `longsum_manual` | 2 -> 1 | 1 -> 0 |
  | `ragged_row_max` | 6 -> 4 | 4 -> 3 |
  | `welford` | 33 -> 32 | 8 -> 7 |
  | `softmax_two_pass` | 14 -> 14 | 4 -> 4 |
  | `hl.reduce` add/max | 4 -> 0 | not measured -> 0 |
  | `cumsum`, `hl.associative_scan` max | 5 -> 2 | not measured -> 0 |

  `layer_norm`, `rms_norm` and `cross_entropy` have no carried loop and are unchanged.
  - `mlir/in_place.py`, run by `execution.inline_module` after inlining. An all-parallel
    `linalg.generic` that is yielded for an iter arg, reads it at the output's indices and
    ignores its own destination takes the iter arg as `outs` and reads it there. Doing this
    after inlining covers torch-mlir helpers too, with no `outs` operand on helpers. If the
    update is `acc op partial`, and `partial` is a reduction used only there that combines
    with the same `op` (`addf`, `mulf`, `maximumf`/`maxnumf`, `minimumf`/`minnumf`) from
    its identity, the reduction starts from `acc` instead and the update is erased. This
    reassociates floating-point sums (`acc + (x1 + x2)` becomes `(acc + x1) + x2`).
  - Reasons for the allocations that remain:
    - An old value read after its update (`exp(mi - mi_next)` in softmax, the old mean in
      Welford) needs a copy.
    - An update that does not read the iter arg directly (`di * exp(...) + sum` adds to
      `mul(di, ...)`) keeps a fresh destination.
    - A padded ragged tile is a buffer (`tensor.pad`), and so is its `_mask_to` select.
    - torch-mlir lowers `amax` to a max-with-index `linalg.generic` with two results, and
      `torch.maximum` to `cmpf`/`select`, so a running max does not match the fold.
    - Temporaries that are not carried are left to the opt pipeline's fusion and
      vectorization. The scalar pipeline is for verification.
  - `hl.reduce` with a single known combiner (`add`, `mul`, `maximum`, `minimum` on floats
    and signed integers, `logical_and`/`logical_or` on bool) is a `linalg.reduce` from the
    op's identity. Other combiners, dtypes (unsigned, `add` on bool) and tuple inputs keep
    Phase 7's lowering.
  - Scans keep the sequential loop. torch-mlir lowers `cumsum`/`cumprod`/`logcumsumexp` to
    `tm_tensor.scan`, which upstream MLIR cannot parse, and fails to import `cummax`
    (`temp/phase8/probe_scan_helpers.py`). The loop already combines a whole slice of the
    other dims per step. Its `take` now uses a rank-reducing `extract_slice` (a view)
    instead of `extract_slice` + `tensor.reshape`, which copied every step. `put` keeps
    `reshape` + `insert_slice`, because a rank-reducing `insert_slice` there trips an MLIR
    assertion in the opt pipeline (see §9).
  - The opt pipeline expands lighthouse's `x86_64/lower.yaml` and adds
    `lower-vector-multi-reduction`. Without it, a vectorized reduction fails to JIT
    (`convert-vector-to-llvm` no longer includes that lowering).
  - Open: the opt-pipeline compile of `layer_norm` (256x1024, 32-row blocks) and of
    `longsum_manual` with a 2048-wide block ran for more than 7 minutes without
    finishing. Not investigated; neither is in the opt test suite.

### Phase 9: Dynamic shapes

**Goal.** `static_shapes=False` kernels compile once per config and run for any sizes in
Helion's shape bucket, instead of failing. Any size that is known at compile time stays a
static dimension; the rest become MLIR `?`. The scalar pipeline handles every dynamic
kernel; the optimizing pipeline handles those whose tile computations are static (a tiled
matmul), and the rest fall back to the scalar pipeline. Static kernels are unchanged.
**Addresses.** I30, I31, I32.

**Feasibility (evaluated before the phase, `temp/phase9/`).**
- Helion (`probe_dynamic_device_ir.py`): host tensors carry size symbols
  (`TensorSizeOrigin`/`NameOrigin`); device IR reads runtime sizes through
  `_get_symnode('x_size1')`, which Phase 4 already passes as a runtime scalar argument, or
  through `sym_size.int` of tile values. The bound kernel is reused for every size in
  Helion's bucket (0/1/≥2 per dim), so one compiled entry must accept all of them.
- Today (I30): a dynamic matmul fails with `DynamicShapeError` (host tensor type).
  Worse, `geometry._static_int_or_none` calls `int()` on `SymInt`s, which specializes them
  to the example sizes behind Helion's back: `row_softmax` compiled for (20, 33) is reused
  by Helion for (21, 40) and rejected by the driver's shape check
  (`probe_row_softmax_dynamic.py`). Any use of size hints must go.
- Lighthouse, scalar pipeline (`spike_dynamic.py scalar`): hand-written modules in the
  backend's form (`?` host tensors, `tensor.dim` sizes, forall trip counts from
  `ceildiv`, `affine.min` real sizes, pad on load, partial stores) give correct results
  from one compiled entry for a matmul (64x64@64x64, 70x100@100x50, 1x7@7x3,
  33x129@129x65), a batch matmul with a dynamic batch, and a row sum with a
  `tensor<8x?xf32>` tile.
- Lighthouse, optimizing pipeline (`spike_dynamic.py opt`): the dynamic matmul is correct
  for all four shapes. The dynamic-batch matmul fails in lighthouse's
  `move_offsets_to_subview` transform, which builds a `memref.subview` with the static
  size sentinel of a dynamic memref (I31); with a two-line guard (skip dynamic memrefs,
  applied locally in `lighthouse/`) it is correct for batch 1, 3 and 8. A reduction over a
  dynamic tile dim fails with "Attempted to vectorize, but failed" (vectorization without
  vector sizes, I32), and tiles < 32 still abort (I23).
- torch-mlir helpers (`spike_dynamic_helpers.py`): with symbolic sample tensors
  (`FakeTensorMode` + `ShapeEnv`, one size symbol per `?`), `amax`, `sum`, `exp`,
  broadcasting `sub`/`div`, `add`, `mm` with a dynamic K and `view` lower to the expected
  signatures (`(tensor<32x?xf32>, tensor<32x1xf32>) -> tensor<32x?xf32>`, ...).
- Most of the machinery exists: Phase 6 already emits dynamic slice sizes, pads, dynamic
  forall regions and `affine.min` against runtime loop ends. Conclusion: feasible; the
  work is resolving sizes to values and letting `?` through the places that assume static
  types.

Work items:
1. **Sizes.** One resolver, `BuildContext.size(expr) -> int | ir.Value`, for every size
   the backend needs (loop bounds, spans, tensor extents, creation shapes):
   - simplify with the config's block sizes and the shape env's replacements; an
     expression without free symbols is an `int`;
   - a size symbol becomes `tensor.dim` of the first host tensor argument whose type has
     it at some dim, or the runtime scalar argument that already carries it
     (`ctx.scalars`);
   - compound expressions (`s0 // 2`, `s0 * s1`) become index arithmetic with Python
     semantics (`scalar_ops`);
   - values are emitted at the entry of the phase function, so they dominate every use;
   - no `int()` on a `SymInt` anywhere (`_static_int_or_none` checks for free symbols
     instead); a symbol with no source raises `DynamicShapeError` naming its origin.
2. **Types.** Host tensor refs get `?` for unresolved sizes (`codegen._tensor_type`, the
   memref entry). `EntryArg.shape` keeps `-1` for them; the driver checks rank, dtype and
   the static dims only, and checks that dims sharing one symbol are equal at run time
   (Helion's duck sizing gives equal example sizes one symbol). `execute_mlir` (no host
   code) rejects host-created tensors of dynamic shape, pointing at `compile_mlir`.
3. **Geometry.** Root and nested loop bounds and spans are expressions resolved through
   item 1; `tile_extent` is the block size when the span is not static; forall trip
   counts are `ceildiv(span, block)` values; `tile.count` likewise.
4. **Slices and creation.** `tile_window` skips the tensor clamp when the loop end and the
   tensor extent are the same expression, else clamps dynamically (as now). A full slice
   of a dynamic dim is a dynamic tile dim (`DimSlice.tile` may be dynamic, no padding).
   `emit.empty/filled/cast_tensor/mask/pad_high` take dynamic dims (sizes from
   `tensor.dim` of their source). `hl.zeros([tile, n])` with a dynamic `n` resolves `n`
   through item 1. `sym_size.int` of a dynamic dim is `tensor.dim`.
5. **Helpers.** `_sample` builds symbolic fake tensors for `?` dims; symbolic result dims
   map to `?`; the request key already contains the `?` types. `infer_results` the same.
   `view`/`reshape` support dynamic dims that map one-to-one (expand/collapse around
   them); other dynamic reshapes raise `UnsupportedOperationError`.
6. **Contractions, transposes, stores.** Dynamic operand dims in `linalg.matmul`/
   `contract`/`transpose` (empties from `tensor.dim`); `_check_accumulator` compares
   static dims only.
7. **Pipeline choice.** The optimizing pipeline is used only if, after inlining, every
   linalg op has static operand shapes (dynamic loop bounds and slice sizes are fine);
   otherwise the scalar pipeline, with a debug log. Carry the lighthouse
   `move_offsets_to_subview` guard (§9) until it is upstream.
8. Static kernels: no `?`, `tensor.dim` or `ceildiv` may appear when every symbol
   resolves (goldens unchanged).

- **New tests** (`tests/test_dynamic_shapes.py`, a `static_shapes=False` kernel per case,
  each called with several shapes including sizes 1 and non-multiples of the block):
  - matmul (`addmm` and `hl.dot`) on the scalar and the optimizing pipeline; one compile
    for all shapes (count compiles);
  - batch matmul with a dynamic batch and static M/N/K on the optimizing pipeline
    (isolated);
  - 1-D and 2-D elementwise, row softmax and layer norm over a dynamic full row, a
    reduction loop over a dynamic K;
  - host-computed bounds (`n = x.size(0) // 2`), `hl.zeros([tile, n])`, `tile.count`,
    `sym_size`, a gather with a dynamic source;
  - `hl.specialize(x.size(1))` keeps that dim static (no `?` in the IR);
  - the `row_softmax` case above: reused for new sizes, never specialized to hints;
  - a dynamic reshape that is not one-to-one raises a clear error.
- **Conformance:** the sweep gets a dynamic mode (`static_shapes=False` for every case) as
  a second baseline column.
- **Exit:** the dynamic sweep column passes every case the static column passes (on the
  scalar pipeline at least); the dynamic matmul is correct on both pipelines; static
  goldens, AMX gate and opt suite unchanged.
- **Risks:** Helion may reuse a bound kernel across sizes in ways the backend cannot see
  (hence the run-time equality check of shared symbols); dynamic kernels mostly run on the
  scalar pipeline until I32 is addressed upstream (masked vectorization with vector
  sizes, or peeling), so their performance is not a goal of this phase.
- **Outcome (done).** Suite 423 -> 449 (`tests/test_dynamic_shapes.py`, 26 tests); static
  sweep 22/22 and the new dynamic sweep column (`--dynamic`,
  `scripts/conformance_baseline_dynamic.json`) 22/22 (16/22 on its first run). Goldens,
  AMX gate and opt suite unchanged. The dynamic matmul (`addmm`, `hl.dot`) is correct on
  both pipelines from one compile for all shapes of a bucket, and so is the dynamic-batch
  matmul on the optimizing pipeline. Compiling `row_softmax` adds no guard to Helion's
  shape env. Deviations and findings:
  - `BuildContext.size` is the resolver. Size values are emitted at the start of the phase
    function, before its first body op, and cached by expression. So a loop end and a
    tensor extent with the same expression are the same SSA value, and `tile_window` skips
    the clamp by value identity. `ctx.extent(value, dim, name)` gives a host tensor's
    runtime extent through it.
  - Geometry keeps spans and root bounds as expressions. A persistent reduction over a
    runtime dim (`x[tm, :]`) is a *whole* block (`BlockGeometry.whole`): one tile of the
    runtime span, instead of Helion's `next_power_of_2(hint)` block. Its block symbol maps
    to the span, so `hl.zeros([tm, n])` and `_get_symnode('block_size_r')` become
    `tensor.dim` values. Nested loop ends that carry a size (`x_size1`) resolve by
    expression rather than through the runtime scalar.
  - Helion does not duck-size: equal example sizes of different inputs get different
    symbols. So the run-time equality check (`driver._check_sizes`) only matters for sizes
    derived in host code (`out = empty([m, n])`, `n // 2`); a loop over one input's extent
    clamps against another input's runtime extent.
  - Dynamic reshapes: in-device `view`/`reshape` that only add or drop unit dims use
    collapse/expand. Other dynamic views fall back to the torch-mlir helper, which lowers
    them (flattening two runtime dims works), so no error is raised as planned. Host views
    of runtime-shaped parameters (`x.view(-1, k)`) collapse to 1-D and expand with
    `ctx.size` values; this fixed the last 6 dynamic sweep cases except `gather_gemv`.
  - `gather_gemv` (`tile.index // S` with a runtime `S`): a runtime scalar where the op's
    schema takes a Tensor now becomes a 0-d tensor operand, since torch-mlir rejects
    `div.Tensor_mode(Tensor, int)`.
  - Helion itself rejects `hl.zeros([tm, n])` reduced over the runtime `n` (an assertion
    in its inductor lowering), so the test stores it instead.
  - Pipeline choice (`execution._dynamic_linalg_op`): the optimizing pipeline is used
    unless a linalg op in the inlined module has a runtime-sized operand or result.
    `row_softmax` then falls back to scalar. The lighthouse `move_offsets_to_subview`
    guard (§9, I31) is still a local patch.
  - Contraction results with runtime dims take their sizes from the operands
    (`x[:, :] @ y[:, tn]`). A contraction over a runtime full K on static output tiles
    stays a `linalg.matmul` with a `?` K.

### Phase 10: Tooling and autotuning

**Goal.** Tunable, cached compilation.
**Addresses.** I26.

Work items:
1. Make pipeline selection a backend config key, with the env var as a fallback.
2. Add an in-process JIT cache keyed by (MLIR text hash, pipeline), optionally on disk via
   `Runner.dump_object_file` (the key then also covers the lighthouse/LLVM versions and
   the CPU features).
3. Implement `MLIRBackend.autotune` with Helion's autotuner and CPU benchmarking of the
   compiled callables. The search space is restricted to tiles ≥ 32 under the opt pipeline
   until the lighthouse fix (I23), and prefers divisible block sizes (no pad/mask work).
   The CPU utilities (`helion_mlir_cpu_utils`) can then drop their hard-coded block sizes.
4. Optional: revisit the attention prototype; online softmax is now legal thanks to Phases
   3, 6 and 7. Profile lighthouse compile time.
- **Outcome (done).** Suite 449 -> 460 (`tests/test_autotune.py`); sweeps 22/22 static and
  dynamic; examples pass. Deviations and findings:
  - Fixed configs everywhere (user request): the 14 test kernels that reached the backend's
    `autotune` (direct calls without `config=`), the two direct-call examples and every
    sweep case (`_BLOCK_SIZES`, Helion's default block sizes for those inputs) now pass a
    config. `tests/conftest.py` and the sweep's subprocesses set
    `HELION_DISALLOW_AUTOTUNING=1`; the tuning tests re-enable it.
  - Pipeline key: `mlir_pipeline` (`"scalar"`/`"opt"`) per config, then
    `HELION_MLIR_PIPELINE`. Helion validates config keys against `VALID_KEYS`, so
    `inject.install` adds it there (monkeypatch, no Helion edit).
    `MLIRBackend.supports_config_key` accepts only `block_sizes` and `mlir_pipeline`: the
    default config and the search space drop the Triton-only keys, and a config with
    another key raises `InvalidConfig`.
  - Autotuning keeps Helion's semantics (`Backend.autotune`). One config is used as is,
    several are searched, and no config means a full search. The backend adds wall-clock
    CPU benchmarking (`do_bench_generic`) and no precompile subprocess. Compile errors skip
    the candidate (`classify_autotune_exception`). Random block sizes are biased 4:1 toward
    divisors of the dimension (`config_value_priors`).
  - Helion's `LocalAutotuneCache` asserts on CPU devices (its key has no CPU runtime). The
    backend swaps in `mlir/autotune.py`'s `CpuAutotuneCache`, keyed by CPU model and
    default pipeline, registered in `helion.autotuner.cache_classes`. Best configs persist
    in `HELION_CACHE_DIR`.
  - Tiles >= 32 under the opt pipeline: a search raises the block minimums (including the
    default config the autotuner uses as its baseline) to `min(32, dim)`. Given configs are
    untouched, so the CPU utilities' packed layouts (`[1, 1, 8, 32]`) keep their sizes.
    The CPU utilities keep their hard-coded configs: tuning on a benchmark's first call
    would take minutes, and a tuned config can be pasted in as before.
  - JIT cache: in-process, LRU of 128 entries keyed by (module text hash, pipeline). It is
    bypassed while an IR dump is enabled. Two configs rarely give the same module (tile
    offsets carry the block size even for a single trip), so hits come from recompiling
    one config (new bound kernels, `compile_mlir`). No on-disk JIT cache: lighthouse's
    `Runner` can dump an object file but not load one.
  - Compile-time profile (`temp/phase10/compile_profile.py`): on the scalar pipeline a
    256x256 matmul or a 32x1024-row softmax compiles in about 0.1-0.3 s per stage. On the
    opt pipeline the matmul takes 0.2 s of lighthouse passes, while the softmax takes 2.5 s
    of lighthouse passes and 7 s of LLVM JIT (wide unrolled vectors). This is the likely
    cause of Phase 8's multi-minute `layer_norm` compile. A quick search of a matmul takes
    about 25 s (scalar) to 50 s (opt).
  - The attention revisit (item 4) was not done.

### Cleanup and stress pass (after Phase 10)

**Outcome (2026-09-29).**
- Modularization: `lowering/loops.py` (root foralls, sequential nests, nested `scf.for`)
  split from `control_flow.py` (`if`/`while`, `_phi`, shared subgraph and carried-value
  helpers); `mlir/sizes.py` `Sizes` (`ctx.sizes`) holds the size resolver that was in
  `BuildContext`; `lowering/__init__` exports only `build_phase_body`/`lower_node`;
  `view_ops.reshape` (was `static_reshape`) and one `emit.reassociation` helper.
- Dead code removed: `TensorEffects.reads/written_in/read_in`, `emit.zero_attr`,
  `torch_tensor_to_mlir_type`; `mlir_dtype_to_torch` no longer defaults to f32.
- Bugs found (by `temp/cleanup/stress.py`: 64 kernels, odd f32 shapes 1..100, 8 block
  size patterns, static and dynamic, 2718 runs; and targeted probes), all fixed:
  - Nested `_for_loop` carried values were the *last* N loop args; a loop reading an
    invariant tile after its accumulator (`acc + x.sum() * scale`) carried the wrong one.
    `_while_loop` assumed every arg is carried (native abort). Both now match outputs to
    variables through `_phi(before, getitem(loop, i))`. Combined `hl.tile([a, b])` inner
    loops may carry values now.
  - A reshape between `?` shapes with equal types but swapped sizes returned its input;
    views now compare size symbols. Runtime scalars at `SymInt` schema args of ATen
    helpers were sampled as 1, giving static helper result types (`1x1x1`).
  - `x[tm, n - 1]` with a runtime `n` was rejected; zero-trip loops (`hl.tile(k // 2)`
    with `k == 1`) divided by zero; `_mask_to` of an Inductor-internal buffer
    (`torch.var` with partial tiles) crashed; a block size Helion specialized to a
    constant (`hl.register_block_size` of a size-1 dim) was unresolvable (static) or read
    as the runtime scalar `n` (dynamic).
  - Opt pipeline: a `linalg.batch_matmul` of padded tiles returned NaNs (lighthouse);
    such modules take the scalar pipeline.
- Tests: `tests/test_numerics.py` plus cases in the control-flow, dynamic-shape and opt
  suites. Suite 484; sweeps 22/22 static and dynamic.

**Second pass (2026-09-29).**
- `aten_bridge/helpers.py` split: `original_args.py` (the Helion patch and the
  restore), `samples.py` (binding, samples, result types; each module's own
  `Sampler`/fake tensor mode instead of one process-wide shape env that grew by
  ~150 symbols per 10 dynamic compiles), `helper_cache.py` (`HelperRequest`,
  torch-mlir run, `HelperCache`); `helpers.py` keeps the call-site API. Keyword
  arguments are matched against the op schema too; `static_dim` moved to support.
- Removed, after coverage over the suite, both sweeps, the 2718-run stress sweep and
  the examples showed them never reached: `subscript_ops.py` (Helion's
  `hl.subscript` only admits `None`/`:`; `lower_subscript` is a unit-dim reshape in
  `view_ops`, the gather moved to loads), `method_ops.py` and the `call_method`
  path (Helion traces tensor methods as ATen ops), `aten.t`/`aten.transpose.int`
  handling (traced as `permute`), the `sym_size` index resolution (covered by symbol
  origins), the tile-scalar block-id fallback, `BuildContext.get_value(float)`, a
  `SymInt` shape branch, an impossible static partial-tile branch, loop-bound and
  `bind` literal fallbacks, and the reverse parameter lookup in host aliases.
- Bugs found on the way: `x[tile.index + k]` crashed (`ir.ops.arith`); a runtime
  offset (`tile.index + shift`) was silently dropped (now rejected); host views that
  are not reshapes (`x[:, 2:]`, `x[::2]`, `x.t()`) were read as their base parameter
  (now inputs of their own); `owned_dims` misaligned index positions after a `None`.
  Suite 499; sweeps 22/22 static and dynamic; stress unchanged.

---

## 6. Issue traceability

| ID | Issue | Phase | Guard / regression test |
|---|---|---|---|
| I1 | `_for_loop` args misread; nonzero `begin` wrong | 1 | nonzero-begin probe |
| I2 | `_for_loop_step` unsupported | 1 | `hl.grid` step probe |
| I3 | Silent sentinels (`-1`, `0`, loop-stack fallback) | 1 | error-path unit tests |
| I4 | Stepped forall blocks SFC remap | 1 | SFC IR check |
| I5 | Substring op matching (`mm`~`addmm`, `div.Tensor_mode`), `mul(x, None)` hack | 2 | `acc + addmm` probe, `x // 3` |
| I6 | `hl.dot` unsupported | 2 | `hl.dot` probe and variants |
| I7 | Three contraction paths; bf16 accumulation without acc | 2 | contraction IR assertions, AMX gate |
| I8 | Errors without source location; misleading hints; broad excepts | 2 | error-message test |
| I9 | Wrappers, lazy imports, duplication, dead code, `BuildContext` mix | 2, 3 | review + suite |
| I10 | `tensor.generate` casts/gathers (not vectorizable) | 2, 5 | goldens |
| I11 | Deferred stores, synthetic accumulator (store-then-load, multi-store, routing, top-level loops) | 3 | storeload, twostores probes |
| I12 | Output resolution heuristics | 3, 4 | multi-output tests |
| I13 | Host code skipped (partial writes, in-place, `out=`, return expressions) | 4 | partial, inplace, outparam probes |
| I14 | Scalar args fail or would be baked in | 4, 5 | scalar-twice probe |
| I15 | Non-contiguous inputs wrong; aliasing vs `restrict` | 4 | noncontig and aliasing tests |
| I16 | `generate_mlir` mutates settings; stateful executor; env-var-only pipeline | 4, 8 | settings test |
| I17 | ATen shape guessing, IR mutation, batch failure (softmax, `torch.where`) | 5 | softmax, where probes |
| I18 | Ragged tiles (crash/NaN), `_mask_to` pass-through, `extra_mask` ignored | 6 | ragged probes, fuzz |
| I19 | `_if`, `_and`/`_or`/`_not`, index comparisons | 7 | if probe |
| I20 | `_while_loop`, `_reduce`, `_associative_scan`, `split`/`join`, `_constant_tensor` | 7 | per-op tests, sweep |
| I21 | Atomics, rand | 7 | explicit-error tests |
| I22 | Lighthouse yield-equivalence (bufferization config) | 0 | unfused-acc spike as a test |
| I23 | Lighthouse small-tile assertion | 0 (avoid), upstream | reproducer (§9) |
| I24 | No harness, isolation, goldens, AMX gate or conformance metric | 0 | — |
| I25 | `uint8` mapped to `ui8` | 2 | dtype test |
| I26 | No autotuning or compile cache | 10 | `tests/test_autotune.py` |
| I27 | Size-1 broadcasting store rejected | 7 | `temp/probe_store_broadcast.py` |
| I28 | Host tensor shape depending on a block size gets a dynamic type | 7 | register_block_size shape test |
| I29 | Host-side Helion API calls (`hl.specialize`) not evaluated in host code | 7 | sweep `matmul_layernorm` |
| I30 | Dynamic shapes fail, or are specialized to the example sizes via `int(SymInt)` | 9 | `tests/test_dynamic_shapes.py`, `conformance_sweep.py --dynamic` |
| I31 | Lighthouse `move_offsets_to_subview` breaks on dynamic memrefs | 9 (local guard), upstream | `test_dynamic_batch_matmul_on_the_optimizing_pipeline` |
| I32 | Optimizing pipeline cannot vectorize linalg ops on dynamic shapes | resolved in the pipeline (masked vectorization, `_compiler/helion_transforms.py`) | `test_runtime_sized_linalg_ops_on_the_optimizing_pipeline` |
| I33 | Per-iteration allocations of carried values; sequential reduce/scan | 8 | `tests/test_in_place_updates.py`, known-combiner tests in `tests/test_language_ops.py` |

---

## 7. Rough size impact

| File | Now | After |
|---|---:|---:|
| `lowering/control_flow.py` | 1214 | about 450 (`ops/loops.py` + `tensor_state.py`) |
| `aten_lowering.py` + `aten_prepass.py` + `helper_rebuild.py` | 1171 | about 250 |
| `codegen.py` | 895 | about 250 |
| `lowering/memory_ops.py` + `for_store_context.py` + `output_resolver.py` | 432 | about 150 |
| `phase_plan.py` + `bound_kernel.py` driver | 414 | about 200 |
| `symbolic_shape_restoration.py` | 163 | 0 (if the audit passes) |
| `matmul_ops.py` + `einsum_ops.py` + contraction parts of `aten_ops.py` | ~550 | about 250 (`ops/contraction.py` + `canonicalize.py`) |

New modules (`geometry`, `tensor_effects`, `signature`, `registry`, `emit`, entry builder) add
roughly 900 LOC.

---

## 8. Risks and open points
- **Bufferization of partially written owned tiles** under tile-and-fuse. The full-tile case
  is verified. The packing kernels in the opt suite cover partial writes.
- **Proving disjoint ownership.** The conservative sequential fallback keeps results correct,
  but costs parallelism, so it is logged.
- **Per-iteration allocations** permitted by `allow-return-allocs-from-loops` for non-DPS
  carried values. This is a performance cost, not a correctness one. Phase 8 made the common
  updates destination-passing; the remaining cases and their reasons are listed in its
  outcome.
- **Normalized forall under tile-and-fuse.** Verified for a contraction. Phase 1 keeps an
  internal switch.
- **Batch torch-mlir failure isolation** multiplies compile time, but only on error paths.
- **`symbolic_shape_restoration`** is removed only after an empirical zero-hit audit.
- **AMX correctness is validated at IR level only** on this machine. Execution on an AMX host
  remains a manual check.

---

## 9. Lighthouse deviations and upstream items

Deviations the backend carries in its own pipeline YAMLs (Phase 0), for later upstreaming:
- One-shot bufferization with `allow-return-allocs-from-loops`. Without it, any loop-carried
  value that is not updated in place fails with "Yield operand #0 is not equivalent to the
  corresponding iter bbArg". With it, the full suite passes 284/284 on the scalar variant.
- The scalar pipeline adds `bufferization-cleanup.yaml` (the deallocation pipeline) because
  loop allocations become possible.
- The optimizing pipeline expands `x86_64/lower.yaml` and adds
  `func.func(lower-vector-multi-reduction)` before the LLVM lowering (Phase 8), without which
  vectorized reductions fail to JIT.

Items to report or fix upstream:
- **Small-tile assertion.** Reproducer: `scripts/lighthouse_small_tile_repro.py N`.
  - Payload: a single `linalg.elementwise <add>` on `tensor<Nxf32>`, run through the backend's
    optimizing pipeline.
  - N < 32 aborts in `applyTilingToAll` ("Mismatched number of loops"); N ≥ 32 completes.
  - Likely fix: skip fusion roots whose annotated tile sizes are all zero in
    `tile_and_fuse_annotated` or `get_fusion_roots`.
  - Until it is fixed, optimizing-pipeline workloads and autotuning use tiles ≥ 32.
- **Dynamic memrefs in `move_offsets_to_subview`** (I31): `create_subview` passes the
  memref type's shape, including the dynamic-size sentinel, as static subview sizes and
  fails ("mixed static/dynamic offset/sizes/strides requires explicit result type").
  Local fix: skip memrefs without a static shape. Ragged tiles of static tensors still
  reach it with dynamic subviews (e.g. a 64x40 elementwise kernel with 32x32 tiles).
  Reproducer: `temp/phase9/spike_dynamic.py opt batch_matmul`.
- **Padded `linalg.batch_matmul`:** the optimizing pipeline returns NaNs for a
  `batch_matmul` whose operands are `tensor.pad`-ed partial tiles; plain `matmul` is
  correct. Reproducer: `scripts/lighthouse_padded_batch_matmul_repro.py batch`.
  Fixed by the opt pipeline's `vectorize_pads` stage (see `docs/MLIR_LIMITATIONS.md`, §15).
- **Vectorization of dynamic shapes** (I32): the schedule vectorizes without vector sizes,
  so a linalg op on a dynamic tile fails ("Attempted to vectorize, but failed"), and
  register unrolling fails on loops of runtime trip count. Resolved in the opt pipeline:
  masked vectorization with the tile bounds as vector sizes, and those loops left rolled.
- **Rank-reducing `insert_slice` in a loop** (Phase 8): with an
  `insert_slice tensor<32xf32> into tensor<32x1024xf32>[0, %i] [32, 1]` carried by an
  `scf.for`, `x86_64/vectorize.yaml` aborts in
  `ValueBoundsConstraintSet::areEquivalentSlices` ("expected slices of same rank").
  Reproducer: drop the `reshape` in `view_ops.put` and run
  `temp/phase8/opt_allocs.py cumsum`. The backend avoids the rank-reducing form there.
- **torch-mlir scans:** `aten.cumsum`/`cumprod`/`logcumsumexp` lower to `tm_tensor.scan`,
  which upstream MLIR (and so lighthouse) cannot parse, and `aten.cummax` fails in the
  FX importer (`NameError: sparsity`). Reproducer: `temp/phase8/probe_scan_helpers.py`.
- **`result_to_args`:** support in/out arguments, returning an argument unchanged, and
  non-tensor scalars. Once available, the backend's own entry can be replaced.
- **Strided function-boundary layouts,** so non-contiguous tensors avoid a copy.
- **SFC remap** only accepts normalized 2-D foralls. The backend adapts in Phase 1; no upstream
  change is required.
- **Known AMX blockers,** already documented in `docs/AMX_MATMUL_OPTIMIZATION_FINDINGS.md`:
  - A transposed inner-block B is silently miscompiled.
  - Rank-5 VNNI contractions fail to vectorize.
  - Outer tiles larger than `[1, 1]` hit an `eraseOp` assertion.
