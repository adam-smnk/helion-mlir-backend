"""Run curated upstream Helion examples through the MLIR backend on CPU.

Each case runs in its own subprocess and is classified pass / wrong / error / crash.

    uv run python scripts/conformance_sweep.py                    # compare to baseline
    uv run python scripts/conformance_sweep.py --update-baseline  # re-record baseline
    uv run python scripts/conformance_sweep.py --case softmax     # run one case inline
    uv run python scripts/conformance_sweep.py --dynamic          # static_shapes=False
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO = Path(__file__).resolve().parents[1]
_BASELINE = Path(__file__).with_name("conformance_baseline.json")
_DYNAMIC_BASELINE = Path(__file__).with_name("conformance_baseline_dynamic.json")
_TIMEOUT_S = 300
# Fixed configs (Helion's default block sizes for these inputs), so the sweep never tunes.
_BLOCK_SIZES = {
    "add": [32, 32],
    "bmm": [1, 16, 16, 16],
    "broadcast_matmul": [16, 16, 16],
    "concat2d_dim1": [32, 32],
    "cross_entropy": [32],
    "embedding": [32, 32],
    "exp": [32],
    "gather_gemv": [32, 32],
    "geglu": [32],
    "layer_norm": [32],
    "longsum": [1],
    "longsum_manual": [32, 4],
    "longsum_w_red_loop": [1],
    "matmul": [16, 16, 16],
    "matmul_layernorm": [32, 32],
    "rms_norm": [32],
    "softmax": [32],
    "softmax_decomposed": [32],
    "softmax_two_pass": [32, 32],
    "sum": [32],
    "swiglu": [32],
    "welford": [16, 16, 16],
}


@dataclass(frozen=True)
class Case:
    module: str
    kernel: str
    inputs: Callable[[], list[object]]
    reference: Callable[..., object]


def _cases() -> dict[str, Case]:
    import torch
    import torch.nn.functional as F

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape)

    def example(name: str) -> object:
        return importlib.import_module(f"examples.{name}")

    return {
        "add": Case("add", "add", lambda: [randn(64, 64), randn(64, 64)], torch.add),
        "exp": Case("exp", "exp_fwd", lambda: [randn(1024)], torch.exp),
        "softmax": Case(
            "softmax", "softmax", lambda: [randn(32, 64)], lambda x: x.softmax(-1)
        ),
        "softmax_decomposed": Case(
            "softmax",
            "softmax_decomposed",
            lambda: [randn(32, 64)],
            lambda x: x.softmax(-1),
        ),
        "softmax_two_pass": Case(
            "softmax",
            "softmax_two_pass",
            lambda: [randn(32, 64)],
            lambda x: x.softmax(-1),
        ),
        "sum": Case("sum", "sum_kernel", lambda: [randn(32, 64)], lambda x: x.sum(-1)),
        "longsum": Case(
            "long_sum", "longsum", lambda: [randn(4, 1024)], lambda x: x.sum(-1)
        ),
        "longsum_w_red_loop": Case(
            "long_sum",
            "longsum_w_red_loop",
            lambda: [randn(4, 1024)],
            lambda x: x.sum(-1),
        ),
        "longsum_manual": Case(
            "long_sum", "longsum_manual", lambda: [randn(4, 1024)], lambda x: x.sum(-1)
        ),
        "rms_norm": Case(
            "rms_norm",
            "rms_norm_fwd",
            lambda: [randn(32, 64), randn(64), 1e-5],
            lambda x, w, eps: example("rms_norm").rms_norm_pytorch(x, w, eps),
        ),
        "layer_norm": Case(
            "layer_norm",
            "layer_norm_fwd",
            lambda: [randn(32, 64), [64], randn(64), randn(64), 1e-5],
            lambda x, shape, w, b, eps: F.layer_norm(x, shape, w, b, eps),
        ),
        "welford": Case(
            "welford",
            "welford",
            lambda: [torch.rand(64), torch.rand(64), torch.rand(32, 64)],
            lambda w, b, x: example("welford").eager_layer_norm(w, b, x),
        ),
        "matmul": Case(
            "matmul", "matmul", lambda: [randn(64, 64), randn(64, 64)], torch.matmul
        ),
        "bmm": Case(
            "bmm", "bmm", lambda: [randn(2, 32, 32), randn(2, 32, 32)], torch.bmm
        ),
        "broadcast_matmul": Case(
            "broadcast_matmul",
            "broadcast_matmul",
            lambda: [randn(2, 32, 64), randn(64, 32)],
            torch.matmul,
        ),
        "matmul_layernorm": Case(
            "matmul_layernorm",
            "matmul_layernorm",
            lambda: [randn(32, 64), randn(64, 64), randn(64), randn(64)],
            lambda x, y, w, b: example("matmul_layernorm").matmul_layernorm_pytorch(
                x, y, w, b
            ),
        ),
        "cross_entropy": Case(
            "cross_entropy",
            "cross_entropy",
            lambda: [randn(32, 128), torch.randint(0, 128, (32,))],
            F.cross_entropy,
        ),
        "embedding": Case(
            "embedding",
            "embedding",
            lambda: [torch.randint(0, 64, (16, 8), dtype=torch.int32), randn(64, 32)],
            F.embedding,
        ),
        "geglu": Case(
            "geglu",
            "_geglu",
            lambda: [randn(32, 64), randn(32, 64)],
            lambda a, b: F.gelu(a, approximate="tanh") * b,
        ),
        "swiglu": Case(
            "swiglu",
            "_swiglu_fwd",
            lambda: [randn(32, 64), randn(32, 64)],
            lambda a, b: F.silu(a) * b,
        ),
        "concat2d_dim1": Case(
            "concatenate",
            "concat2d_dim1",
            lambda: [randn(32, 16), randn(32, 48)],
            lambda x, y: torch.cat([x, y], dim=1),
        ),
        "gather_gemv": Case(
            "gather_gemv",
            "gather_gemv",
            lambda: [
                randn(4, 32, 32),
                torch.randint(0, 4, (8,), dtype=torch.int32),
                randn(32),
            ],
            lambda w, idx, x: torch.stack([w[i] @ x for i in idx.tolist()]),
        ),
    }


def _run_case_inline(name: str, dynamic: bool = False) -> str:
    """Execute one case in this process; returns a one-line classification."""
    sys.path.insert(0, str(_REPO / "helion"))
    import helion
    import torch

    import helion_mlir_backend  # noqa: F401

    case = _cases()[name]
    torch.manual_seed(0)
    source = getattr(importlib.import_module(f"examples.{case.module}"), case.kernel)
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=not dynamic,
        config=helion.Config(block_sizes=_BLOCK_SIZES[name]),
    )(source.fn)
    args = case.inputs()
    try:
        actual = kernel(*args)
    except Exception as exc:
        message = str(exc).strip().splitlines()
        return f"error {type(exc).__name__}: {message[0][:160] if message else ''}"
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
    expected = case.reference(*args)
    if tuple(actual.shape) != tuple(expected.shape):
        return f"wrong shape={tuple(actual.shape)} expected={tuple(expected.shape)}"
    try:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    except AssertionError:
        return f"wrong max_err={(actual.float() - expected.float()).abs().max().item():.3g}"
    return "pass"


def _run_case_subprocess(name: str, dynamic: bool) -> dict[str, str]:
    try:
        completed = subprocess.run(
            [sys.executable, __file__, "--case", name]
            + (["--dynamic"] if dynamic else []),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            cwd=_REPO,
            check=False,
            env={**os.environ, "HELION_DISALLOW_AUTOTUNING": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "detail": f"timeout after {_TIMEOUT_S}s"}
    if completed.returncode < 0:
        return {"status": "crash", "detail": f"signal {-completed.returncode}"}
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith("RESULT ")
    ]
    if not lines:
        tail = (completed.stderr.strip().splitlines() or ["no output"])[-1]
        return {"status": "error", "detail": tail[:200]}
    status, _, detail = lines[-1].removeprefix("RESULT ").partition(" ")
    return {"status": status, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run a single case in this process")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--dynamic", action="store_true", help="static_shapes=False (own baseline)"
    )
    options = parser.parse_args()

    if options.case:
        print(f"RESULT {_run_case_inline(options.case, options.dynamic)}", flush=True)
        return 0

    baseline_path = _DYNAMIC_BASELINE if options.dynamic else _BASELINE
    results = {
        name: _run_case_subprocess(name, options.dynamic) for name in sorted(_cases())
    }
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    regressions = []
    for name, result in results.items():
        previous = baseline.get(name, {}).get("status")
        marker = ""
        if previous == "pass" and result["status"] != "pass":
            marker = "  <-- REGRESSION"
            regressions.append(name)
        elif previous not in (None, "pass") and result["status"] == "pass":
            marker = "  <-- fixed"
        print(f"{result['status']:<6} {name:<22} {result['detail']}{marker}")
    counts = dict.fromkeys(("pass", "wrong", "error", "crash"), 0)
    for result in results.values():
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(", ".join(f"{status}={count}" for status, count in counts.items()))

    if options.update_baseline:
        baseline_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"baseline written to {baseline_path}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
