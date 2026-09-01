"""Authoritative (block_id, bias, is_scalar) resolution for index nodes.

Single source of truth for mapping a Helion device-IR index expression to the
block id / bias it represents, built entirely from Helion's own device-IR
metadata (``tile_with_offset``, symbol origins) instead of name-based or
id()-keyed heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .block_ids import SCALAR_SYMBOL_KINDS
from .block_ids import block_id_from_key

if TYPE_CHECKING:
    from ..build_context import BuildContext


@dataclass(frozen=True)
class IndexDescriptor:
    """Resolved identity of a single subscript index position."""

    block_id: int | None
    bias: int
    is_scalar: bool


_UNRESOLVED = IndexDescriptor(block_id=None, bias=0, is_scalar=False)


def resolve_index_descriptor(ctx: BuildContext, index_node: object) -> IndexDescriptor:
    """Resolve an index expression to its block id, bias, and scalar-ness.

    Resolution order (all authoritative, no name/heuristic matching):
    1. Literal int -> scalar constant offset.
    2. ``meta['tile_with_offset']`` -> block id + offset, set for every
       backend by Helion's own ``add_tile_with_offset_metadata`` pass.
    3. Symbol origin (``HostFunction.expr_to_origin`` via
       ``BuildContext.symbol_info``) -> block id, plus whether it denotes a
       scalar grid/tile position or a tile extent.
    4. ``_get_symnode('block_size_N')`` key -> block id directly.
    5. ``sym_size.int(tensor, dim)`` -> the referenced tensor dimension's own
       symbol origin.
    """
    import torch.fx

    if isinstance(index_node, int):
        return IndexDescriptor(block_id=None, bias=index_node, is_scalar=True)
    if not isinstance(index_node, torch.fx.Node):
        return _UNRESOLVED

    tile_meta = index_node.meta.get("tile_with_offset")
    if tile_meta is not None:
        block_id = tile_meta.get("block_id")
        offset = tile_meta.get("offset", 0)
        bias = offset if isinstance(offset, int) else 0
        return IndexDescriptor(block_id=block_id, bias=bias, is_scalar=False)

    symbol_info = ctx.symbol_info(index_node.meta.get("val"))
    if symbol_info is not None:
        block_id, kind = symbol_info
        return IndexDescriptor(
            block_id=block_id, bias=0, is_scalar=kind in SCALAR_SYMBOL_KINDS
        )

    target_name = getattr(index_node.target, "__name__", "")
    if target_name == "_get_symnode" and index_node.args:
        block_id = block_id_from_key(index_node.args[0])
        if block_id is not None:
            return IndexDescriptor(block_id=block_id, bias=0, is_scalar=False)

    if target_name in ("sym_size.int", "sym_size_int") and len(index_node.args) >= 2:
        tensor_node, dimension_index = index_node.args[0], index_node.args[1]
        if isinstance(tensor_node, torch.fx.Node) and isinstance(dimension_index, int):
            tensor_value = tensor_node.meta.get("val")
            if isinstance(tensor_value, torch.Tensor) and (
                0 <= dimension_index < len(tensor_value.shape)
            ):
                dimension_info = ctx.symbol_info(tensor_value.shape[dimension_index])
                if dimension_info is not None:
                    block_id, kind = dimension_info
                    return IndexDescriptor(
                        block_id=block_id,
                        bias=0,
                        is_scalar=kind in SCALAR_SYMBOL_KINDS,
                    )

    return _UNRESOLVED
