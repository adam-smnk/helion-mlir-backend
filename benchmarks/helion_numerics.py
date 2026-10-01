"""Shared numerical checks for the bf16 Helion benchmarks.

Accuracy is judged against an f64 reference rather than against PyTorch
directly. A bf16 result differs from PyTorch by up to half an ulp purely from a
different accumulation order, so "is it bit-identical to PyTorch" is the wrong
question; "is it *less accurate* than PyTorch" is the right one.

For a deterministic gate, :func:`assert_bitwise_equal` is used on small shapes
built so the exact result is representable in bf16, which removes rounding from
the comparison entirely and leaves layout or indexing bugs as the only way to
fail.
"""

from __future__ import annotations

import torch
from torch import Tensor


def report_accuracy(name: str, actual: Tensor, reference_f64: Tensor) -> None:
    """Print error of *actual* against an f64 ground truth."""
    delta = actual.double() - reference_f64
    error = delta.abs()
    print(
        f"{name:22s} max {error.max().item():.4e}  mean {error.mean().item():.4e}  "
        f"bias {delta.mean().item():+.3e}"
    )


def assert_no_worse_than(
    name: str,
    actual: Tensor,
    baseline: Tensor,
    reference_f64: Tensor,
    margin: float = 1.05,
) -> None:
    """Assert *actual* is no less accurate than *baseline* against *reference_f64*.

    Only the max error is asserted. Mean-error ratios are reported but not
    enforced: at bf16 output resolution the mean is decided by a handful of
    discrete rounding disagreements, and across chained layers those legitimately
    compound, so a ratio on it is noise rather than signal. Bit-exactness is
    covered by :func:`assert_bitwise_equal` on small shapes instead.
    """
    actual_error = (actual.double() - reference_f64).abs()
    baseline_error = (baseline.double() - reference_f64).abs()
    actual_max, actual_mean = actual_error.max().item(), actual_error.mean().item()
    base_max, base_mean = baseline_error.max().item(), baseline_error.mean().item()
    worse = (actual_error > baseline_error).sum().item()
    better = (actual_error < baseline_error).sum().item()
    scale = reference_f64.abs().mean().item()
    # One ulp of the output dtype at this data scale; below it nothing is resolvable.
    ulp = torch.finfo(actual.dtype).eps * scale
    print(
        f"{name:22s} max {actual_max:.4e} vs {base_max:.4e}, "
        f"mean {actual_mean:.4e} vs {base_mean:.4e} ({actual_mean / max(ulp, 1e-300):.3f} "
        f"vs {base_mean / max(ulp, 1e-300):.3f} ulp), worse/better {worse}/{better}"
    )
    if actual_max > base_max * margin and actual_max > ulp:
        raise AssertionError(
            f"{name}: max error {actual_max:.4e} exceeds baseline "
            f"{base_max:.4e} by more than {margin:g}x (1 ulp = {ulp:.3e})"
        )


def assert_bitwise_equal(name: str, actual: Tensor, expected: Tensor) -> None:
    """Assert exact equality; only valid when the true result is representable."""
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype {actual.dtype} != {expected.dtype}")
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    mismatches = (actual != expected).sum().item()
    if mismatches:
        worst = (actual.double() - expected.double()).abs().max().item()
        raise AssertionError(
            f"{name}: {mismatches}/{actual.numel()} elements differ, max delta {worst:g}"
        )
    print(f"{name:22s} bitwise exact ({actual.numel()} elements)")


def signed_permutation(size: int, generator: torch.Generator) -> Tensor:
    """A weight matrix with one ``+-1`` per row, so layer magnitudes cannot grow."""
    weight = torch.zeros(size, size)
    columns = torch.randperm(size, generator=generator)
    signs = torch.randint(0, 2, (size,), generator=generator) * 2 - 1
    weight[torch.arange(size), columns] = signs.to(torch.float32)
    return weight


def small_integer_matrix(
    rows: int, cols: int, generator: torch.Generator, limit: int = 1
) -> Tensor:
    """Integer entries in ``[-limit, limit]``, exactly representable in bf16."""
    values = torch.randint(-limit, limit + 1, (rows, cols), generator=generator)
    return values.to(torch.float32)
