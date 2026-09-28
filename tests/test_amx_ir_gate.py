"""AMX IR gate: production bf16 contractions must reach AMX ops in the optimizing pipeline.

Runs on any x86 host: AMX is forced on through lighthouse's ``TargetInfo.override`` and
the pipeline is applied stage by stage until ``x86.amx`` ops appear. Nothing is executed.
"""

from __future__ import annotations

from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline.driver import BackendDriver
from mlir import ir
from mlir.passmanager import PassManager
import pytest

from tests.harness import blocked_matmul_cases

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.execution import pipeline_descriptor

_AMX_FEATURES = ["amx_bf16", "amx_tile", "amx_int8"]


def _first_amx_ops(module: ir.Module, entry: str) -> set[str]:
    features = TargetInfo.host().features + _AMX_FEATURES
    with TargetInfo.override(features=features), module.context, ir.Location.unknown():
        PassManager.parse("builtin.module(inline,canonicalize)").run(module.operation)
        driver = BackendDriver(module, entry, result_to_args=True, benchmark=False)
        driver.add_stage(pipeline_descriptor(optimized=True))
        for stage in driver.stages:
            module = stage.apply(module)
            if "x86" not in str(stage):
                continue
            ops = {
                token.split("(")[0]
                for token in str(module).split()
                if token.startswith("x86.amx.")
            }
            if ops:
                return ops
    return set()


@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(
    "case", ["blocked_matmul", "blocked_matmul_bias", "blocked_matmul_affine"]
)
def test_blocked_bf16_matmul_reaches_amx(case: str) -> None:
    kernel, args = blocked_matmul_cases()[case]
    ops = _first_amx_ops(generate_mlir(kernel, args), kernel.fn.__name__)
    assert "x86.amx.tile_mulf" in ops, (
        f"no AMX contraction produced (saw {sorted(ops)})"
    )
