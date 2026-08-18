"""Dispatch Helion-specific FX nodes to MLIR lowering methods."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

import torch.fx

if TYPE_CHECKING:
    import mlir.ir as ir

    from ..codegen import MLIRModuleBuilder


Lowerer = Callable[[torch.fx.Node], object]


def lower_helion_node(
    builder: MLIRModuleBuilder,
    node: torch.fx.Node,
    target_name: str,
) -> tuple[bool, ir.Value | None]:
    """Lower a Helion tracing node, returning whether it was recognized."""
    from ..lowering.memory_ops import lower_getitem

    lowerers: dict[str, Lowerer] = {
        "_host_tensor": builder._lower_host_tensor,
        "_get_symnode": builder._lower_get_symnode,
        "_new_var": lambda current: builder._get_value(current.args[0]),
        "_phi": lambda current: builder._get_value(current.args[1]),
        "_for_loop": builder._lower_for_loop,
        "getitem": lambda current: lower_getitem(builder.context, current),
        "load": builder._lower_load,
        "store": builder._lower_store_node,
        "full": builder._lower_full,
        "zeros": builder._lower_zeros,
        "sym_size.int": builder._lower_sym_size,
        "sym_size_int": builder._lower_sym_size,
        "tile_index": builder._lower_tile_index,
        "_mask_to": builder._lower_mask_to,
    }
    lowerer = lowerers.get(target_name)
    if lowerer is None:
        return False, None
    return True, lowerer(node)
