"""Runtime backend registration helpers for external Helion MLIR backend."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def install() -> bool:
    """Register external MLIR backend into Helion registry.

    Returns True when registration is applied successfully.
    """
    try:
        from helion._compiler.backend_registry import register_compiler_backend
        from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
    except Exception as exc:
        log.debug("External MLIR backend registration unavailable: %s", exc)
        return False

    register_compiler_backend(MLIRBackend)
    return True
