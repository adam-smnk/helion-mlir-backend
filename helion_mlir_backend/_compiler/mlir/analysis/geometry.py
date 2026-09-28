"""Authoritative per-block-id loop geometry for one compiled config.

Everything here comes from Helion's own metadata, never from tensor shapes:
- the block size of each block id from the config (``BlockSizeInfo.from_config``; for an
  ``hl.grid`` loop this is its step),
- the iteration span from ``BlockSizeInfo.size`` (``end - begin``),
- tile vs grid kind from the loop target's type (``TileIndexType`` / ``GridIndexType``),
- the ``begin``/``end`` of every top-level loop from its ``hl.tile``/``hl.grid`` call.

Nested loops carry their own bounds on the ``_for_loop(graph_id, begin, end, args)`` /
``_for_loop_step(..., step)`` node, and their block ids on ``ForLoopGraphInfo.block_ids``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import torch
import torch.fx

from ..support.errors import DynamicShapeError

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction

LoopKind = Literal["tile", "grid"]
LOOP_TARGET_NAMES = ("_for_loop", "_for_loop_step")


@dataclass(frozen=True)
class BlockGeometry:
    block_id: int
    block_size: int
    kind: LoopKind
    span: int | None
    """Static ``end - begin``, or ``None`` when the loop bounds are runtime values."""

    @property
    def tile_extent(self) -> int:
        """Static size of one iteration's slice (``hl.grid`` dims are indexed as scalars)."""
        if self.span is None:
            return self.block_size
        return min(self.block_size, self.span)


@dataclass(frozen=True)
class LoopBounds:
    """Bounds of one loop dimension: ints, or FX nodes for runtime values."""

    begin: int | torch.fx.Node
    end: int | torch.fx.Node
    step: int


@dataclass
class KernelGeometry:
    blocks: dict[int, BlockGeometry]
    root_bounds: dict[int, LoopBounds]

    @classmethod
    def from_host_function(
        cls, hf: HostFunction, config: object, env: CompileEnvironment
    ) -> KernelGeometry:
        """Build the geometry; must run with ``hf`` active (``with hf:``)."""
        kinds, root_calls = _loop_kinds_and_root_calls(hf)
        blocks: dict[int, BlockGeometry] = {}
        for info in env.block_sizes:
            configured = info.from_config(config)
            block_size = _static_int(
                configured if configured is not None else info.size,
                f"block size of block_id {info.block_id}",
            )
            blocks[info.block_id] = BlockGeometry(
                block_id=info.block_id,
                block_size=block_size,
                kind=kinds.get(info.block_id, "tile"),
                span=_static_int_or_none(info.size),
            )
        root_bounds: dict[int, LoopBounds] = {}
        for block_ids, (begins, ends) in root_calls:
            for block_id, begin, end in zip(block_ids, begins, ends, strict=True):
                root_bounds[block_id] = LoopBounds(
                    begin=_static_int(begin, f"begin of block_id {block_id}"),
                    end=_static_int(end, f"end of block_id {block_id}"),
                    step=blocks[block_id].block_size,
                )
        return cls(blocks=blocks, root_bounds=root_bounds)

    def block(self, block_id: int) -> BlockGeometry:
        return self.blocks[block_id]

    def block_size(self, block_id: int) -> int:
        return self.blocks[block_id].block_size

    def tile_extent(self, block_id: int) -> int:
        return self.blocks[block_id].tile_extent

    def is_grid(self, block_id: int) -> bool:
        return self.blocks[block_id].kind == "grid"

    def block_sizes(self) -> dict[int, int]:
        return {bid: block.block_size for bid, block in self.blocks.items()}

    def spans(self) -> dict[int, int]:
        """Static spans only (block ids with runtime bounds are omitted)."""
        return {
            bid: block.span
            for bid, block in self.blocks.items()
            if block.span is not None
        }

    def trip_count(self, block_id: int) -> int:
        block = self.blocks[block_id]
        if block.span is None:
            raise DynamicShapeError(
                block_id, symbol_name=f"span of block_id {block_id}"
            )
        return -(-block.span // block.block_size)

    def is_ragged(self, block_id: int) -> bool:
        block = self.blocks[block_id]
        return (
            block.kind == "tile"
            and block.span is not None
            and block.span % block.block_size != 0
        )

    def loop_bounds(
        self, node: torch.fx.Node, block_ids: list[int]
    ) -> list[LoopBounds]:
        """Bounds of a ``_for_loop``/``_for_loop_step`` node, one per block id."""
        begins, ends = list(node.args[1]), list(node.args[2])
        steps: list[int | None] = (
            list(node.args[4])
            if len(node.args) > 4 and node.args[4] is not None
            else [None] * len(block_ids)
        )
        return [
            LoopBounds(
                begin=begin,
                end=end,
                step=step if step is not None else self.block_size(block_id),
            )
            for block_id, begin, end, step in zip(
                block_ids, begins, ends, steps, strict=True
            )
        ]


def is_loop_node(node: object) -> bool:
    return getattr(getattr(node, "target", None), "__name__", "") in LOOP_TARGET_NAMES


def loop_block_ids(hf: HostFunction, node: torch.fx.Node) -> list[int]:
    """The block ids a ``_for_loop``/``_for_loop_step`` node iterates, in order."""
    return list(hf.device_ir.graphs[node.args[0]].block_ids)


def _static_int(value: object, what: str) -> int:
    resolved = _static_int_or_none(value)
    if resolved is None:
        raise DynamicShapeError(value, symbol_name=what)
    return resolved


def _static_int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _loop_kinds_and_root_calls(
    hf: HostFunction,
) -> tuple[dict[int, LoopKind], list[tuple[list[int], tuple[list, list]]]]:
    """Tile/grid kind of every loop block id, and each top-level loop's begins/ends."""
    from helion._compiler.ast_extension import LoopType
    from helion._compiler.type_info import GridIndexType
    from helion._compiler.type_info import IterType
    from helion._compiler.type_info import SequenceType

    kinds: dict[int, LoopKind] = {}
    root_calls: list[tuple[list[int], tuple[list, list]]] = []
    for node in ast.walk(ast.Module(body=list(hf.body), type_ignores=[])):
        if not isinstance(node, ast.For) or getattr(node, "_loop_type", None) not in (
            LoopType.GRID,
            LoopType.DEVICE,
        ):
            continue
        iter_type = getattr(node.iter, "_type_info", None)
        if not isinstance(iter_type, IterType):
            continue
        inner = iter_type.inner
        index_types = inner.unpack() if isinstance(inner, SequenceType) else [inner]
        block_ids = [index_type.block_id for index_type in index_types]
        for index_type in index_types:
            kinds[index_type.block_id] = (
                "grid" if isinstance(index_type, GridIndexType) else "tile"
            )
        if node._loop_type == LoopType.GRID:
            root_calls.append((block_ids, _root_begins_ends(node, len(block_ids))))
    return kinds, root_calls


def _root_begins_ends(node: ast.For, rank: int) -> tuple[list, list]:
    """Evaluate a top-level ``hl.tile``/``hl.grid`` call's begin/end proxies."""
    call = node.iter
    args = [arg._type_info.proxy() for arg in call.args]
    first = args[0]
    second = args[1] if len(args) > 1 else None
    if second is None:
        begins, ends = [0] * rank, first
    else:
        begins, ends = first, second
    as_list = lambda value: list(value) if isinstance(value, (list, tuple)) else [value]  # noqa: E731
    return as_list(begins), as_list(ends)
