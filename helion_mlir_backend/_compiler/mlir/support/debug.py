"""Environment-driven debug options, read in one place."""

from __future__ import annotations

from dataclasses import dataclass
import os

_FALSE_VALUES = ("", "0", "false", "no")
PIPELINE_CONFIG_KEY = "mlir_pipeline"
"""Config key naming the lighthouse pipeline (``scalar``/``opt``) of one config."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSE_VALUES


@dataclass(frozen=True)
class DebugOptions:
    """IR dump switches (``HELION_MLIR_DUMP_*``)."""

    dump_ir: bool
    dump_pre_lowering: bool
    dump_lowered: bool

    @classmethod
    def from_env(cls) -> DebugOptions:
        # Read per call: tests and users toggle these at runtime.
        return cls(
            dump_ir=_flag("HELION_MLIR_DUMP_IR"),
            dump_pre_lowering=_flag("HELION_MLIR_DUMP_PRE_LOWERING"),
            dump_lowered=_flag("HELION_MLIR_DUMP_LOWERED"),
        )


def use_optimizing_pipeline() -> bool:
    """Whether ``HELION_MLIR_PIPELINE=1`` selects the optimizing pipeline."""
    return os.environ.get("HELION_MLIR_PIPELINE", "").strip() == "1"
