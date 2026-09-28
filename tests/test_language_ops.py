"""Helion language ops lowered directly (plan Phase 7): ``torch.tensor`` constants,
``hl.reduce``, ``hl.associative_scan``/``torch.cumsum`` and ``hl.split``/``hl.join``."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


def max_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def keep_first(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Non-commutative: exposes the combine order of a reverse scan."""
    return a * 0.0 + a + b * 0.5


@_kernel(8)
def constant_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * torch.tensor(2.0)
    return out


@_kernel(4)
def reduce_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(max_combine, x[tile, :], dim=1)
    return out


@_kernel(4)
def cumsum_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = torch.cumsum(x[tile, :], dim=1)
    return out


@_kernel(4)
def reverse_scan_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = hl.associative_scan(keep_first, x[tile, :], dim=1, reverse=True)
    return out


@_kernel(4)
def swap_pairs_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a, b = hl.split(x[tile, :].reshape([tile, 4, 2]))
        out[tile, :] = hl.join(b, a).reshape([tile, 8])
    return out


@_kernel(4)
def atomic_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [0], x[tile].sum())
    return out


def _reverse_scan_reference(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    acc = x[:, -1]
    out[:, -1] = acc
    for i in range(x.size(1) - 2, -1, -1):
        acc = keep_first(x[:, i], acc)
        out[:, i] = acc
    return out


def test_constant_tensor() -> None:
    torch.manual_seed(0)
    check_kernel(constant_kernel, lambda x: x * 2.0, [torch.randn(16)])


def test_reduce_with_combine_function() -> None:
    torch.manual_seed(0)
    check_kernel(reduce_kernel, lambda x: x.amax(dim=1), [torch.randn(8, 16)])


def test_cumsum() -> None:
    torch.manual_seed(0)
    check_kernel(
        cumsum_kernel,
        lambda x: torch.cumsum(x, dim=1),
        [torch.randn(8, 16)],
        atol=1e-5,
        rtol=1e-5,
    )


def test_reverse_scan() -> None:
    torch.manual_seed(0)
    check_kernel(reverse_scan_kernel, _reverse_scan_reference, [torch.randn(8, 6)])


def test_split_and_join() -> None:
    torch.manual_seed(0)
    check_kernel(
        swap_pairs_kernel,
        lambda x: x.view(8, 4, 2).flip(-1).reshape(8, 8),
        [torch.randn(8, 8)],
    )


def test_atomics_are_rejected_with_a_reason() -> None:
    with pytest.raises(UnsupportedOperationError, match="atomics need memory"):
        generate_mlir(atomic_kernel, [torch.randn(16)])
