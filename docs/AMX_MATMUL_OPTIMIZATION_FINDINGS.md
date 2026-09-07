# AMX Blocked Matmul: Optimization Findings and Future Work

Three optimizations for the bf16 blocked matmul were prototyped and are recorded
here because each is currently blocked in the backend rather than in kernel
authoring. All three are numerically correct under the scalar pipeline, so the
kernels below are valid starting points: they should begin working as the
corresponding backend gap closes.

Context: the Helion MLIR backend uses its own pipeline
(`helion_mlir_backend/_compiler/pipeline.yaml`), which deliberately omits
lighthouse's `block_pack_matmuls`. Packing is the kernel author's job, and the
layout the AMX stages expect is

```text
A = [MB, KB, BM, BK]        C(blocked) = [MB, NB, BM, BN]
B = [NB, KB, BK, BN]        C(merged)  = [MB, BM, NB, BN]  -> viewable as [M, N]
```

`helion_matmul_bf16.py` already produces exactly this. `BM = BN = BK = 32`.

Reproduce with the probes in `temp/`:

```bash
# scalar (correctness)
env -u HELION_MLIR_PIPELINE OMP_NUM_THREADS=4 uv run python temp/<probe>.py
# AMX (lowering + performance)
HELION_MLIR_PIPELINE=1 OMP_NUM_THREADS=4 uv run python temp/<probe>.py
```

---

## 1. VNNI Block-Packed Contraction

**Status:** frontend works, AMX vectorization fails.
**Probe:** `temp/probe_vnni_blocked_contract.py`

### Summary

A contraction whose operands are both block-packed *and* VNNI-packed lowers
correctly through the frontend and is numerically correct under the scalar
pipeline (max err `1.562e-02` at 256). Under `HELION_MLIR_PIPELINE=1` it fails in
lighthouse vectorization with `Attempted to vectorize, but failed`, before AMX
conversion is ever attempted.

### Explanation

The shape that matters is lighthouse's rank-5 form (see
`lighthouse/lighthouse/ingress/mlir_gen/generic.py`, the `rank == 5` case):

```text
A = [MB, KB, BM, BK/V, V]
B = [NB, KB, BK/V, BN, V]
C = [MB, BM, NB, BN]
```

with `V = 2` for bf16. As an einsum this is `"akmcv,bkcnv->abmn"`, a contraction
with **three** reduction dims (`k`, `c`, `v`). That the frontend accepts this is
a recent improvement: direct `torch.einsum` capture maps it onto a single
`linalg.contract` instead of decomposing it.

An earlier round of testing used a flat 3-D layout (`[M, K/V, V]` with
`"mkv,knv->mn"`) and concluded VNNI was unsupported. That was the wrong shape for
a block-packed pipeline; the rank-5 form above is the one to pursue.

Note the operand preparation is asymmetric:

* **A is free.** `[MB, KB, BM, BK]` splits its contiguous `BK` into `(BK/V, V)`
  as a pure view.
* **B needs a real pack.** `[NB, KB, BK, BN]` must become
  `[NB, KB, BK/V, BN, V]`, which transposes the innermost two axes.

### Key improvement opportunities

* Teach lighthouse vectorization to handle rank-5 contraction operands. This is
  the single blocker; AMX conversion is never reached.
* Once vectorized, this path may avoid the online-packing limitation entirely:
  `VectorContractToAMXDotProduct.cpp` has a distinct `isVnni` code path that does
  not need to synthesize the VNNI layout at runtime.
* Fold the B VNNI pack into the existing block pack so the operand is produced in
  one pass rather than two.

### Minimal example

```python
VNNI = 2

@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def vnni_blocked_matmul(a5: Tensor, b5: Tensor) -> Tensor:
    """[MB, KB, BM, BK/V, V] x [NB, KB, BK/V, BN, V] -> [MB, BM, NB, BN]."""
    blocks_m, blocks_k, block_m, block_kv, vnni = a5.shape
    blocks_n, blocks_k2, block_kv2, block_n, vnni2 = b5.shape
    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n), dtype=a5.dtype, device=a5.device
    )
    for tm, tn in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros([tm, tn, block_m, block_n], dtype=torch.float32)
        acc = acc + torch.einsum(
            "akmcv,bkcnv->abmn", a5[tm, :, :, :, :], b5[tn, :, :, :, :]
        )
        out[tm, :, tn, :] = acc.to(a5.dtype).permute(0, 2, 1, 3)
    return out


# Operand prep from the already block-packed a4 / b4:
a5 = a4.view(nb, nb, BLOCK, BLOCK // VNNI, VNNI)                 # free view
b5 = (
    b4.view(nb, nb, BLOCK // VNNI, VNNI, BLOCK)
    .permute(0, 1, 2, 4, 3)
    .contiguous()                                                 # real pack
)
```

---

## 2. Split-K With Preallocated Partials

**Status:** frontend works, AMX lowering fails. Also net-negative on traffic for
square shapes.
**Probe:** `temp/probe_split_k.py`

### Summary

Split the reduction into `S` slabs, write each slab's partial product into its
own slice of a preallocated `[S, MB, BM, NB, BN]` f32 buffer, then reduce once.
This avoids reloading an accumulator per slab. Numerically correct under the
scalar pipeline (max err `1.562e-02`). Under AMX it leaves a `vector.contract`
behind, producing `builtin.unrealized_conversion_cast` and
`LLVM Translation failed` / `Failure while creating the ExecutionEngine`.

### Explanation

Expressed as a two-phase kernel: phase one contracts per slab with the slab index
as an extra *parallel* dimension (`"askmc,bskcn->sambn"`), phase two sums over
`S`. The extra parallel dim inside the contraction is what defeats vectorization;
the plain 4-D form without it vectorizes fine.

Two things are worth knowing before investing here:

* **The traffic accounting is unfavourable for square shapes.** At 4096, split-K
  adds `S x 64 MB` of partial writes plus `S x 64 MB` of reduction reads, against
  at most 128 MB saved. It only pays off when `K` dominates `M` and `N`
  (tall-skinny), which is not the benchmark shape.
* **Multi-phase kernels have a hard structural rule:** every `torch.empty` must
  be hoisted *above* the first top-level loop. Only `hl.barrier()` may appear
  between top-level loops, otherwise Helion raises
  `TopLevelStatementBetweenLoops`.

### Key improvement opportunities

* Allow a parallel batch-like dimension in a contraction that is otherwise
  AMX-shaped, so the slab index does not block vectorization.
* Revisit for tall-skinny shapes, where the extra partial traffic is amortized by
  a much larger `K`.
* If the goal is parallelism rather than traffic, note the blocked matmul already
  exposes `MB x NB` (16384 at 4096) independent tiles, so split-K adds little.

### Minimal example

```python
@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 1, 1, 32, 1, 32]),
)
def split_k_matmul(a5: Tensor, b5: Tensor) -> Tensor:
    """[MB, S, KBs, BM, BK] x [NB, S, KBs, BK, BN] -> [MB, BM, NB, BN]."""
    blocks_m, slabs, slab_k, block_m, block_k = a5.shape
    blocks_n, slabs2, slab_k2, block_k2, block_n = b5.shape

    # Every allocation must precede the first top-level loop.
    partial = torch.empty(
        (slabs, blocks_m, block_m, blocks_n, block_n),
        dtype=torch.float32,
        device=a5.device,
    )
    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n), dtype=a5.dtype, device=a5.device
    )

    for tm, tn, ts in hl.tile([blocks_m, blocks_n, slabs]):
        acc = hl.zeros([ts, tm, block_m, tn, block_n], dtype=torch.float32)
        acc = acc + torch.einsum(
            "askmc,bskcn->sambn", a5[tm, ts, :, :, :], b5[tn, ts, :, :, :]
        )
        partial[ts, tm, :, tn, :] = acc

    hl.barrier()

    for mb, tile_bm, nb, tile_bn in hl.tile(
        [blocks_m, block_m, blocks_n, block_n]
    ):
        out[mb, tile_bm, nb, tile_bn] = (
            partial[:, mb, tile_bm, nb, tile_bn].sum(0).to(a5.dtype)
        )
    return out


# Operand prep: split the KB axis of the packed operands into (S, KBs).
a5 = a4.view(nb, SLABS, nb // SLABS, BLOCK, BLOCK)
b5 = b4.view(nb, SLABS, nb // SLABS, BLOCK, BLOCK)
```

---

## 3. Cache Blocking via Larger Outer Tiles

**Status:** blocked by a compiler assertion. Likely the highest-leverage of the
three.
**Probe:** `temp/probe_block_tile_sweep.py` (set `TMB`/`TNB`; run one config per
process, the assertion aborts)

### Summary

The blocked matmul compiles **only** with `block_sizes=[1, 1]`. Every larger
outer tile crashes the compiler:

```text
block_sizes=[1,1]   0.925 ms   2322.4 GFLOP/s
block_sizes=[1,2]   COMPILER ASSERTION
block_sizes=[2,1]   COMPILER ASSERTION
block_sizes=[2,2]   COMPILER ASSERTION
block_sizes=[4,4]   COMPILER ASSERTION
```

```text
PatternMatch.cpp:181 eraseOp: Assertion
`mayBeGraphRegion(*op->getParentRegion()) && "expected that op has no uses"'
```

### Explanation

With `[1, 1]` each loop iteration computes a single `32 x 32` output block and
consumes a full `K` panel of A and of B. Nothing is reused *between* iterations
at the kernel level, so every A panel is re-read once per N block and every B
panel once per M block. Nominal read traffic is

```text
MB * NB * (K * BM + K * BN) * 2 bytes = 128 * 128 * 4096 * 64 * 2 ~= 8.6 GB
```

at 4096. In practice most of that is absorbed by the hardware cache hierarchy,
but it means reuse is entirely at the mercy of cache capacity rather than being
scheduled. oneDNN, by contrast, does explicit multi-level cache blocking, which
is part of why it wins end to end even though our matmul-only throughput is
competitive.

Larger tiles (`[2, 2]`, `[4, 4]`) would compute a `64 x 64` or `128 x 128` output
block per iteration and reuse each loaded panel across the tile, cutting panel
re-reads by the tile factor.

### Key improvement opportunities

* Fix the `eraseOp` assertion. Unlike the other two items this is a plain crash,
  not a missing feature, so it is probably the cheapest to resolve and the most
  directly valuable.
* Once larger tiles compile, sweep `block_sizes` and the accumulator footprint
  together: the f32 accumulator is `tile_mb * tile_nb * BM * BN * 4` bytes and
  should stay L2-resident.
* Consider a separate cache-tile level above the register tile, mirroring
  lighthouse's `strategy=cache` then `strategy=register_*` staging.

### Minimal example

```python
def make_blocked_matmul(tile_mb: int, tile_nb: int):
    """Only tile_mb == tile_nb == 1 currently compiles."""

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[tile_mb, tile_nb]),
    )
    def merged(a4: Tensor, b4: Tensor) -> Tensor:
        blocks_m, blocks_k, block_m, block_k = a4.shape
        blocks_n, blocks_k2, block_k2, block_n = b4.shape
        out = torch.empty(
            (blocks_m, block_m, blocks_n, block_n),
            dtype=a4.dtype,
            device=a4.device,
        )
        for tm, tn in hl.tile([blocks_m, blocks_n]):
            acc = hl.zeros([tm, tn, block_m, block_n], dtype=torch.float32)
            acc = acc + torch.einsum(
                "akmc,bkcn->abmn", a4[tm, :, :, :], b4[tn, :, :, :]
            )
            out[tm, :, tn, :] = acc.to(a4.dtype).permute(0, 2, 1, 3)
        return out

    return merged
```

---

## Related Backend Constraints

Two further constraints found while investigating the above; they bound what any
future design can assume.

* **Transposed inner-block B silently miscompiles.** A contraction with B packed
  as `[NB, KB, BN, BK]` (`"akmc,bknc->abmn"`) is correct under the scalar
  pipeline but produces wrong results under AMX, with no error: 95-98% of
  elements wrong, max err `44.4` where `0.5` is expected. The frontend IR is
  byte-identical between the two pipelines. Root cause is in
  `VectorContractToAMXDotProduct.cpp`: `validateContractOps` checks only operand
  *shapes* and never inspects `contractOp.getIndexingMaps()`, so a transposed
  orientation passes validation and is then lowered as if it were the standard
  one. A map check that bails out would turn a silent miscompile into correct
  fallback code.

* **In-loop (BLIS-style) packing cannot be expressed.** Packing a block inside
  the contraction loop, so packed data never round-trips to memory, fails to
  lower. AMX tile loads are emitted by tracing each `vector.contract` operand back
  to a memref read:

  ```cpp
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>([&](auto readOp) { srcBuff = readOp.getOperand(0); });
  return srcBuff && isa<MemRefType>(srcBuff.getType());
  ```

  An in-loop `.permute()` yields a pure SSA vector value, so the trace fails and
  the surviving `vector.contract` cannot be translated. Consequently **AMX
  operands must already be in memory in their final layout**, which is why
  packing is materialized today. Teaching the pattern to spill a non-memref
  operand to a small scratch buffer would unblock this. Probe:
  `temp/probe_blis_inloop_packing.py`.
