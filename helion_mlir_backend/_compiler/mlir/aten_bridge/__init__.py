"""ATen operation bridge for custom and generic lowering."""

from __future__ import annotations

from .aten_helper_table import AtenHelperTable
from .aten_ops import aten_target_matches
from .aten_ops import is_custom_aten
from .aten_ops import lower_custom_aten
from .aten_ops import lower_max_reduce_from_tensor
from .aten_ops import lower_passthrough
from .helper_rebuild import rebuild_aten_helper_for_call
from .torch_mlir_pipeline import batch_import_and_lower

__all__ = [
    "AtenHelperTable",
    "aten_target_matches",
    "batch_import_and_lower",
    "is_custom_aten",
    "lower_custom_aten",
    "lower_max_reduce_from_tensor",
    "lower_passthrough",
    "rebuild_aten_helper_for_call",
]
