# Helion Kernel Authoring Guide

This is a practical manual for writing Helion kernels that are predictable,
portable across Helion backends, and suitable for the MLIR backend in this
repository. It focuses on the mistakes that are easiest to make when a kernel
looks mathematically simple but its tiled value shapes or loop geometry do not
match the destination layout.

The short version:

1. Make every stored value have the same logical axis order as its destination
   slice.
2. Use `hl.tile()` for tunable blocks and `hl.grid()` for non-tunable scalar
   iteration.
3. Treat `block_sizes` as an ordered list of `hl.tile()` loops only.
4. Keep tensor work inside device loops and use explicit barriers for phases.
5. Test numerical results, not only whether compilation or MLIR verification
   succeeds.

For backend-specific support and limitations, see `docs/MLIR_USAGE.md` and
`docs/MLIR_LIMITATIONS.md`.

## 1. Start With the Shape Contract

Before writing the loop, write down the source and destination shapes and the
axis meaning of every index. For example:

```text
source: [K, Panels, BN]
out:    [Panels, K, BN]
```

There are two fundamentally different operations one might intend:

- **Reindexing without changing the value layout:** load
  `source[tile_k, tile_panel, tile_n]` and store into a destination indexed in
  that same order.
- **A layout-changing transpose:** load a value with axes `[K, Panels, BN]`
  and store it as `[Panels, K, BN]`.

The second operation requires an explicit permutation:

```python
value = source[tile_k, tile_panel, tile_n].permute(1, 0, 2)
out[tile_panel, tile_k, tile_n] = value
```

Do not assume that assignment syntax turns the first expression into the
second. `out[a, b] = source[b, a]` is not automatically equivalent to
`source[b, a].permute(1, 0)`.

### The store invariant

For a normal tiled store, the value's dimensions must match the destination
slice's dimensions in order:

```text
value shape:       [tile_m, tile_n]
destination slice: [tile_m, tile_n]
```

This is not enough:

```text
value shape:       [tile_n, tile_m]
destination slice: [tile_m, tile_n]
```

Even if `tile_m == tile_n` for one test shape, the code still has the wrong
axis contract. Use `.permute(...)` explicitly or change the indexing order.

The MLIR backend rejects many mismatched stores with
`UnsupportedOperationError("store with transposed or mismatched tile layout")`
instead of producing invalid `tensor.parallel_insert_slice` operations.

## 2. Use the Right Loop Primitive

### `hl.tile()` means a tunable block

Use `hl.tile()` when the loop represents a block whose size should be selected
by the Helion configuration or autotuner:

```python
for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
```

A one-dimensional tile is also common:

```python
for tile_k in hl.tile(k):
    ...
```

### `hl.grid()` means scalar iteration

Use `hl.grid()` for ordinary scalar iteration over an extent, especially when
the dimension is a small panel, batch, channel, or other non-tiled index:

```python
for panel in hl.grid(panels):
    for tile_k in hl.tile(k):
        for n_idx in hl.grid(bn):
            out[panel, tile_k, n_idx] = source[tile_k, panel, n_idx]
```

A `hl.grid()` loop is not a tunable tile loop. It consumes no entry in
`helion.Config(block_sizes=...)`.

### Nested loop order should follow data movement

Choose a loop order that makes the source and destination index order obvious.
For the copy above, the source and destination are intentionally indexed in
matching value order. If the output layout differs from the source layout,
make the permutation visible in the value expression.

Do not add extra nested loops merely to expose scalar indices if a single
combined tile already expresses the operation clearly. Every additional loop
is another opportunity for block IDs, shapes, and masks to disagree.

## 3. Configure Block Sizes Correctly

`block_sizes` is positional and corresponds to `hl.tile()` loops in traversal
order. It does not include `hl.grid()` loops.

For this kernel:

```python
for panel in hl.grid(panels):       # no config slot
    for tile_k in hl.tile(k):       # block_sizes[0]
        for n_idx in hl.grid(bn):   # no config slot
            ...
```

use:

```python
config=helion.Config(block_sizes=[BK])
```

Not this:

```python
config=helion.Config(block_sizes=[1, BK, BN])
```

The latter assigns `1` to `tile_k`, which can produce a correct but needlessly
slow loop with one-element slices.

For a combined tile:

```python
for tile_m, tile_n in hl.tile([m, n]):
    ...
```

provide two entries, in that order:

```python
config=helion.Config(block_sizes=[BM, BN])
```

When inspecting performance, print or dump the generated loop steps. A
surprising step size is often a configuration-position error, not a lowering
problem.

## 4. Prefer Simple, Shape-Preserving Tile Expressions

The MLIR backend has the most predictable behavior when a loaded tensor slice,
its operations, and its store all preserve the intended tile shape.

Good elementwise pattern:

```python
@helion.kernel(static_shapes=True, backend="mlir")
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out
```

Good tiled matmul pattern:

```python
@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[BM, BN, BK]),
)
def matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + torch.matmul(a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
```

Keep the reduction accumulator's shape equal to the output tile shape. Make the
reduction loop explicit and verify that the operands use compatible inner
axes: `[tile_m, tile_k] @ [tile_k, tile_n]`.

### Contractions written as `torch.einsum`

A two-operand `torch.einsum` is lowered directly to a single `linalg.contract`
whenever the equation fits that op's semantics. This keeps the contraction
intact instead of letting PyTorch expand it into a
`permute`/`view`/`bmm`/`view`/`permute` chain, so transposes and batch axes
become indexing maps rather than data movement:

```python
for tile_m, tile_n in hl.tile([m, n]):
    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    for tile_k in hl.tile(k):
        acc = acc + torch.einsum(
            "mk,nk->mn", a[tile_m, tile_k], b[tile_n, tile_k]
        )
    out[tile_m, tile_n] = acc
```

An equation qualifies when it has exactly two operands, no ellipsis, no
repeated subscript within a single operand or in the output, every output
subscript present in at least one input, and every contracted subscript
present in *both* inputs. Multiple contracted dimensions are fine
(`"mkl,kln->mn"` contracts `k` and `l` in one op). Reduction-free equations
(`"ij,ij->ij"`, `"m,n->mn"`) deliberately keep their elementwise lowering.

Broadcasting comes in two forms and they behave differently:

- *Omitting* a subscript from one operand is supported and free — it simply
  drops that dimension from the operand's indexing map. `"bmk,kn->bmn"`
  broadcasts a shared 2-D matrix across the batch with no data movement.
- Relying on a *size-1* dimension to broadcast against a larger one
  (`"mk,kn->mn"` with `k=1` on one side) is **not** taken by this path.
  `linalg.contract` requires both operands to agree on a contracted extent, so
  the equation is left to PyTorch's decomposition. Prefer an explicit
  `expand`/`broadcast_to`, or omit the subscript instead.

A shared subscript backed by a tile extent is matched by *symbol*, so
`x[tile_m, tile_k]` against `y[tile_k, tile_n]` is recognised as one
contraction without the compiler having to evaluate the tile size.

Equations outside that set — three or more operands, a single operand, a
diagonal like `"kk,kn->kn"`, or a dimension summed out of only one operand
like `"mkl,kn->mn"` — silently fall back to PyTorch's decomposition. That
fallback is still correct, just less direct, and some of the resulting op
chains hit other backend limitations. If you need predictable codegen, prefer
an equation from the qualifying set or write the contraction as
`torch.matmul`.

## 5. Understand Reshape, Broadcast, and Permute

These operations are not interchangeable:

- `reshape` changes how a contiguous sequence of elements is grouped.
- `broadcast` expands size-one dimensions without changing the underlying
  element sequence.
- `permute` changes axis order.

Use the operation that describes the intended transformation. In particular,
do not use a reshape to repair a transposed value. A reshape from
`[tile_k, tile_p]` to `[tile_p, tile_k]` has the same number of elements but does
not have transpose semantics.

For a host-side view or reshape used inside a device loop, preserve the shape
change explicitly and test it numerically:

```python
reshaped = x.view(16, 8)
for _ in hl.tile(16):
    out[:, :] = reshaped
```

The MLIR backend materializes supported static host-side aliases. Dynamic or
incompatible shape changes may still require rewriting the kernel so the
transformation occurs in a supported device-side operation.

## 6. Combined Tiles and Ragged Boundaries

A combined multi-dimensional tile such as:

```python
for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = ...
```

uses a statically shaped slice for each outer iteration in the MLIR backend.
For dimensions that require multiple iterations, choose block sizes that evenly
divide the corresponding extents. Otherwise the final partial tile may not be
representable by the current `scf.forall` slice geometry.

A single-dimensional `hl.tile()` has more flexible ragged-tile handling. If a
combined tile does not divide cleanly, consider:

- choosing a divisor block size;
- moving one dimension to a separate loop; or
- restructuring the operation so only one dimension is tiled together.

The backend reports unsupported ragged combined tiles instead of silently
clamping an invalid static slice.

## 7. Static Shapes and Kernel Structure

For the current MLIR backend:

- author kernels with `@helion.kernel(static_shapes=True)`;
- compile with concrete input tensor shapes;
- annotate tensor arguments;
- put device tensor work inside `hl.tile()` or `hl.grid()` loops;
- define output tensors with explicit shape, dtype, and device;
- keep the kernel source in a real Python file so Helion can inspect it.

A small amount of host-side setup is useful and supported, such as computing a
scalar or tensor before the first device loop. Arbitrary host-style tensor
execution is not a replacement for tiled device code.

Use nontrivial random inputs when testing. Zeros, ones, equal dimensions, and
symmetric matrices can hide an axis-order error.

## 8. Multi-Phase Kernels

Use `hl.barrier()` when a later top-level device loop consumes data written by
a previous loop:

```python
for tile_m, tile_n in hl.tile([m, n]):
    mid[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

hl.barrier()

for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = mid[tile_m, tile_n] * 2.0
```

Rules for the current backend:

- `hl.barrier()` must separate dependent phases;
- no unrelated statement may appear between the two top-level device loops;
- a host-computed tensor needed by a later phase should be computed before the
  first loop that uses it;
- multi-phase and host-tensor interop currently use the direct
  `backend="mlir"` call path;
- phase outputs and the final return should use plain variable names rather
  than arbitrary expressions.

Think of each barrier-separated region as a separate kernel function with an
explicit host-side handoff. Name intermediate tensors clearly so that the
handoff and final return are easy to follow.

## 9. Pipeline Selection and Performance Experiments

The normal scalar pipeline is the baseline for correctness and general kernel
experiments. `HELION_MLIR_PIPELINE=1` selects the AMX vectorizing pipeline in
this repository. It is benchmark-oriented and is not a general compatibility
mode; a kernel that works with the scalar pipeline may abort or fail in the AMX
pipeline.

When investigating a failure, first record:

```bash
printf 'HELION_MLIR_PIPELINE=%s\n' "${HELION_MLIR_PIPELINE-<unset>}"
```

Then reproduce with the variable unset:

```bash
env -u HELION_MLIR_PIPELINE uv run python your_repro.py
```

Only compare performance after confirming that both runs produce the same
numerical result. A slow kernel may have a wrong block-size configuration even
when its MLIR is valid.

## 10. Packing and Vectorized Transposes

Packing is a useful example of a kernel where source-level tile shape affects
the cost of a later vectorized operation. For a layout change from
`[K, Panels, BN]` to `[Panels, K, BN]`, a working pattern is:

```python
@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 8, 32]),
)
def pack_panels(source: torch.Tensor) -> torch.Tensor:
    k, panels, bn = source.shape
    out = torch.empty((panels, k, bn), dtype=source.dtype, device=source.device)
    for panel, tile_k, tile_n in hl.tile([panels, k, bn]):
        out[panel, tile_k, tile_n] = source[tile_k, panel, tile_n].permute(
            1, 0, 2
        )
    return out
```

The unit panel tile is intentional. The `K` sub-tile of `8` and contiguous
`BN` tile of `32` make each transpose input `[8, 1, 32]` and output
`[1, 8, 32]`. With the current vectorizing pipeline, this lowers to a
`vector<256xbf16>` transpose temporary and retains `vector<32xbf16>` loads and
stores.

Using a `K` tile of `32` instead produces a `vector<1024xbf16>` transpose
temporary. It is mathematically equivalent, but creates a much larger shuffle
sequence. In the tested `512 x 512` BF16 workload, `K=8` was faster than `K=16`
and `K=32` despite requiring more loop iterations.

The alternative spelling below avoids an explicit transpose by making the
panel index scalar:

```python
for panel in hl.grid(panels):
    for tile_k, tile_n in hl.tile([k, bn]):
        out[panel, tile_k, tile_n] = source[tile_k, panel, tile_n]
```

This does lower to `vector<32xbf16>` loads and stores, but it was slower in the
tested pipeline because it introduces nested OpenMP work and per-panel
temporary handling. Avoiding a `linalg.transpose` in the source is therefore
not automatically faster.

### Measure enough work

Small packing workloads are dominated by fixed costs: JIT compilation and
cache setup on the first call, OpenMP launch/scheduling, tensor temporaries,
and slice insertion. Compare warmed-up calls and use a workload large enough
to amortize those costs. In this environment, the `512 x 512` case showed a
large gap to native PyTorch, while `4096 x 4096` reached approximately native
bandwidth at both four and eight threads, within normal run-to-run variation.

For post-pipeline inspection, set:

```bash
HELION_MLIR_DUMP_LOWERED=1 HELION_MLIR_PIPELINE=1 \
  OMP_NUM_THREADS=4 uv run python your_kernel.py
```

Look for the actual vector element counts, not only the presence of the word
`vector`. For this packing pattern, `vector<32xbf16>` load/store operations are
expected, while the transpose temporary should be kept small enough for the
target register and shuffle hardware.

## 11. A Disciplined Validation Loop

For every new kernel, validate in this order:

1. **Shape check:** write down source slice shape, operation result shape, and
   destination slice shape for one tile.
2. **Numerical check:** compare against a simple PyTorch reference using random,
   nonsymmetric values.
3. **Boundary check:** test at least one shape with multiple tiles and one shape
   that exercises a partial or awkward boundary, subject to combined-tile
   divisibility rules.
4. **Configuration check:** confirm the number and order of `block_sizes` entries
   matches only the `hl.tile()` loops.
5. **IR check:** inspect loop steps, `tensor.extract_slice` sizes, and
   `tensor.parallel_insert_slice` sizes.
6. **Performance check:** benchmark only after the first five checks pass.

For this repository, useful commands are:

```bash
uv run pytest tests/test_mlir_execution.py -q --log-level=WARNING
uv run ruff check helion_mlir_backend/ tests/
```

To inspect pre-lowering graphs and loop decisions, enable:

```bash
HELION_MLIR_DUMP_PRE_LOWERING=1 uv run python your_repro.py
```

For generated MLIR, use `generate_mlir()` and print the module, or capture the
pre-lowering dump around a direct kernel call.

## 12. Common Symptoms and Likely Causes

| Symptom | Likely cause | First check |
| --- | --- | --- |
| Abort only with `HELION_MLIR_PIPELINE=1` | AMX benchmark pipeline does not support the pattern | Unset the variable and retry |
| Correct output but unexpectedly slow | Wrong positional `block_sizes` entry, often because `hl.grid()` was counted | Inspect generated loop steps |
| Small copy is far slower than native PyTorch | Fixed JIT/OpenMP/tensor overhead dominates the workload | Warm up and repeat at a larger shape |
| Large transpose temporary in lowered IR | Transpose tile is too large for the vectorization path | Reduce the transposed tile dimension and inspect vector widths |
| `output tile larger than the destination dimension` | Wrong loop/block ID resolution or mismatched tile geometry | Compare loop extent and output slice sizes |
| `store with transposed or mismatched tile layout` | Value axes and destination axes are in different orders | Add an explicit `.permute(...)` or align indexing |
| Segmentation fault during insert/store | Invalid slice geometry reached low-level code | Reproduce with current guards and inspect insert sizes |
| Failure only on a partial combined tile | Combined tile does not evenly divide an iterated extent | Choose a divisor block size or split the loops |
| Phase dependency error | Missing `hl.barrier()` between dependent top-level loops | Add the barrier immediately between loops |
| Source inspection error | Kernel was defined dynamically or in an unavailable source context | Move it to a real `.py` file |
| Correct shape assertion but wrong values | Test data is symmetric or the axis order is wrong | Use random nonsymmetric tensors and `torch.testing.assert_close` |

## 13. Authoring Checklist

Before considering a kernel complete, confirm:

- [ ] Every input and output axis has a written meaning.
- [ ] Every loaded tile's axes match the destination slice axes, or the value
      uses an explicit `.permute(...)`.
- [ ] `hl.tile()` is used for tunable blocks and `hl.grid()` for scalar loops.
- [ ] `block_sizes` contains exactly one entry per `hl.tile()` loop, in order.
- [ ] Combined tile dimensions divide cleanly wherever multiple iterations are
      required.
- [ ] Reduction accumulators match the output tile shape.
- [ ] Dependent phases are separated by `hl.barrier()`.
- [ ] Host-side views and reshapes have been tested for both shape and values.
- [ ] Tests use random nonsymmetric data and a PyTorch reference.
- [ ] Correctness was tested with the scalar pipeline before benchmarking.
- [ ] Generated loop steps and extract/insert slice sizes were inspected for a
      new or nontrivial pattern.
- [ ] Post-pipeline vector widths and transpose temporary sizes were inspected
    for performance-sensitive packing kernels.
- [ ] Performance was compared after warm-up on a workload large enough to
    amortize JIT, OpenMP, and tensor-management overhead.

## Related Documentation

- `docs/MLIR_USAGE.md` -- setup, supported examples, and execution paths.
- `docs/MLIR_LIMITATIONS.md` -- current backend limitations and diagnostics.
- `docs/BACKEND_SHAPE_INFERENCE_AND_PROPAGATION.md` -- shape propagation
  details.
- `examples/` -- runnable MLIR backend kernels.
