"""Property-based fuzz coverage for nested-tile/grid indexing geometry.

Targets the exact failure class that has twice caused real segfaults in this
backend (block-id/offset resolution across combined tiles, nested reductions,
and grid+tile reordering) by fuzzing shapes and block sizes across the known
structurally-distinct kernel patterns, instead of relying only on a fixed set
of hand-picked shapes.

A combined-2D-tile + nested-reduction-accumulator fuzz test (varying m/n/k
and block sizes for a tiled matmul) found and fixed two real bugs (block-id
collision in codegen.py's upper-bound pre-pass; a missing upper-bound clamp
in build_context.shape_from_nodes) and then root-caused a third, deeper
architectural gap: the outer combined-tile ``scf.forall`` has no dynamic
per-iteration clamp for a ragged (non-evenly-divisible) boundary tile, and
previously crashed with heap corruption instead of failing cleanly.
``build_kernel_body`` now raises a clear ``UnsupportedOperationError`` for
that case instead -- see repo/session memory for the full investigation.
"""

from __future__ import annotations

import helion
import helion.language as hl
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import torch

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.support.errors import ModuleBuilderError
from helion_mlir_backend._compiler.mlir.support.errors import UnsupportedOperationError

# Derandomized + bounded: deterministic across CI runs, small enough to keep
# per-run compile cost (each example does a full MLIR compile) reasonable.
_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    derandomize=True,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)

_BLOCK_SIZES = [4, 8, 16, 32]


@given(
    m=st.sampled_from([8, 16, 32, 48]),
    n=st.sampled_from([8, 16, 32, 48]),
    k=st.sampled_from([8, 16, 32]),
    bm=st.sampled_from(_BLOCK_SIZES),
    bn=st.sampled_from(_BLOCK_SIZES),
    bk=st.sampled_from(_BLOCK_SIZES),
)
@_SETTINGS
def test_combined_tile_matmul_random_shapes(
    m: int, n: int, k: int, bm: int, bn: int, bk: int
) -> None:
    """Combined 2D tile + nested reduction across randomized shapes/blocks.

    No constraint on block sizes vs dimensions: a ragged (non-evenly-
    divisible, multi-iteration) combined-tile dimension is an accepted,
    cleanly-diagnosed limitation (asserted below), not a crash; everything
    else must compile and execute to numerical parity with eager torch.
    """

    @helion.kernel(static_shapes=True)
    def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m_, k_ = x.shape
        k2_, n_ = y.shape
        out = torch.zeros((m_, n_), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m_, n_]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k_):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = acc
        return out

    torch.manual_seed(0)
    x = torch.randn(m, k)
    y = torch.randn(k, n)
    config = helion.Config(block_sizes=[bm, bn, bk])
    ragged = (bm < m and m % bm != 0) or (bn < n and n % bn != 0)
    try:
        module = generate_mlir(mm, [x, y], config=config)
    except (UnsupportedOperationError, ModuleBuilderError) as exc:
        assert ragged, f"unexpected compile failure for a non-ragged case: {exc}"
        assert "ragged" in str(exc).lower()
        return
    actual = MLIRBackend().execute_mlir(module, x, y, kernel_name="mm")
    torch.testing.assert_close(actual, x @ y, atol=1e-3, rtol=1e-3)


@given(
    n_panels=st.sampled_from([2, 3, 4]),
    m=st.sampled_from([8, 16, 24]),
    bn=st.sampled_from([4, 8, 16]),
)
@_SETTINGS
def test_reordered_store_random_shapes(n_panels: int, m: int, bn: int) -> None:
    """Grid+tile reordered store across randomized shapes/block sizes.

    This exact pattern (``out[tm, panel, :] = ...``) previously segfaulted
    from an exact-set-vs-subset block-id matching bug; fuzz it broadly.
    """

    @helion.kernel(static_shapes=True)
    def unpack_panels(src: torch.Tensor) -> torch.Tensor:
        n_panels_, m_, bnn = src.shape
        out = torch.empty((m_, n_panels_, bnn), dtype=src.dtype, device=src.device)
        for panel in hl.grid(n_panels_):
            for tm in hl.tile(m_):
                out[tm, panel, :] = src[panel, tm, :]
        return out

    torch.manual_seed(1)
    src = torch.randn(n_panels, m, 8)
    config = helion.Config(block_sizes=[1, bn])
    module = generate_mlir(unpack_panels, [src], config=config)
    actual = MLIRBackend().execute_mlir(module, src, kernel_name="unpack_panels")
    torch.testing.assert_close(actual, src.permute(1, 0, 2).contiguous())
