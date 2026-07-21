"""External MLIR backend package for Helion.

This package hosts the MLIR backend implementation outside the Helion tree and
registers an external MLIR backend with vanilla Helion when imported.
"""

from __future__ import annotations

from .api import generate_mlir as generate_mlir
from .inject import install as install

__all__ = ["generate_mlir", "install"]

# Import-time side effect: make backend name "mlir" available in Helion.
install()
