"""Ragged (boundary) tiles: a tile that extends past its loop or tensor (plan Phase 6).

Loads zero-pad the part outside, ``_mask_to`` replaces it by the reduction
identity, and stores write only the real part. When the block size divides the
extent the IR has no dynamic sizes.
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import opt_pipeline
from tests.harness import run_direct

from helion_mlir_backend import generate_mlir


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


@_kernel(8, 16)
def add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + y[tm, tn]
    return out


@_kernel(4, 16)
def row_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@_kernel(8)
def row_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        shifted = torch.exp(row - row.amax(dim=-1, keepdim=True))
        out[tm, :] = shifted / shifted.sum(dim=-1, keepdim=True)
    return out


@_kernel(8)
def chunk_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    """Softmax over each tile of rows: padded rows must not count (``_mask_to``)."""
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        rows = x[tm, :]
        shifted = torch.exp(rows - rows.amax(dim=0, keepdim=True))
        out[tm, :] = shifted / shifted.sum(dim=0, keepdim=True)
    return out


@_kernel(16)
def offset_scale_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(3, x.size(0)):
        out[tile] = x[tile] * 2.0
    return out


@_kernel(32)
def read_past_end_kernel(a: torch.Tensor, n_pad: hl.constexpr) -> torch.Tensor:
    out = torch.empty((int(n_pad),), dtype=a.dtype, device=a.device)
    for tile in hl.tile(int(n_pad)):
        out[tile] = a[tile]
    return out


@_kernel(16)
def masked_load_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.load(x, [tile], extra_mask=tile.index % 2 == 0)
    return out


@_kernel(16)
def masked_store_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, -1.0)
    for tile in hl.tile(x.size(0)):
        hl.store(out, [tile], x[tile], extra_mask=tile.index % 2 == 0)
    return out


@_kernel(32, 32)
def scale_2d_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 3.0
    return out


@_kernel(16)
def store_then_load_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1.0
        out[tile] = x[tile] * 2.0
    return out


@_kernel(32, 32, 32)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


def _every_other(x: torch.Tensor, fill: float) -> torch.Tensor:
    out = torch.full_like(x, fill)
    out[::2] = x[::2]
    return out


def test_combined_2d() -> None:
    torch.manual_seed(0)
    check_kernel(add_2d_kernel, torch.add, [torch.randn(20, 36), torch.randn(20, 36)])


def test_ragged_reduction_loop() -> None:
    torch.manual_seed(0)
    check_kernel(row_sum_kernel, lambda x: x.sum(dim=-1), [torch.randn(10, 20)])


def test_row_softmax_ragged_rows() -> None:
    torch.manual_seed(0)
    check_kernel(
        row_softmax_kernel, lambda x: torch.softmax(x, -1), [torch.randn(20, 16)]
    )


def test_softmax_over_ragged_tile() -> None:
    torch.manual_seed(0)
    x = -torch.rand(20, 6) - 1.0  # all negative, so zero padding would win amax

    def reference(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.softmax(chunk, 0) for chunk in x.split(8)])

    check_kernel(chunk_softmax_kernel, reference, [x])


def test_nonzero_begin() -> None:
    torch.manual_seed(0)
    x = torch.randn(40)

    def reference(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[3:] = x[3:] * 2.0
        return out

    check_kernel(offset_scale_kernel, reference, [x])


def test_read_past_tensor_end_is_zero() -> None:
    """``docs/PADDING_FUSION_FINDINGS.md``, finding 2: the excess reads as zeros."""
    torch.manual_seed(0)
    a = torch.randn(19)
    result = run_direct(read_past_end_kernel, [a, hl.constexpr(64)])
    expected = torch.zeros(64)
    expected[:19] = a
    torch.testing.assert_close(result, expected)


def test_extra_mask_on_load() -> None:
    torch.manual_seed(0)
    check_kernel(masked_load_kernel, lambda x: _every_other(x, 0.0), [torch.randn(40)])


def test_extra_mask_on_store() -> None:
    torch.manual_seed(0)
    check_kernel(
        masked_store_kernel, lambda x: _every_other(x, -1.0), [torch.randn(40)]
    )


def test_divisible_extent_has_static_ir() -> None:
    ir = str(generate_mlir(scale_2d_kernel, [torch.randn(64, 96)]))
    assert "tensor.pad" not in ir
    assert "affine.min" not in ir
    assert "?" not in ir


def test_read_after_write_of_partial_region() -> None:
    torch.manual_seed(0)
    check_kernel(store_then_load_kernel, lambda x: (x + 1.0) * 2.0, [torch.randn(40)])


@pytest.mark.isolated
def test_opt_pipeline() -> None:
    torch.manual_seed(0)
    x = torch.randn(70, 100)
    y = torch.randn(100, 50)
    with opt_pipeline():
        scaled = scale_2d_kernel(x)
        product = matmul_kernel(x, y)
    torch.testing.assert_close(scaled, x * 3.0)
    torch.testing.assert_close(product, x @ y, atol=1e-3, rtol=1e-3)
