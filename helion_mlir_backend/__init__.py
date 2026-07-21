"""External MLIR backend package for Helion.

This package hosts the MLIR backend implementation outside the Helion tree and
registers an external MLIR backend with vanilla Helion when imported.
"""

from .api import generate_mlir as generate_mlir
from .inject import install as install


def _auto_enable() -> None:
    """Enable external MLIR backend registration on import."""
    install()


_auto_enable()
