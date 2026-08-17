"""Shared mutable state for MLIR code generation."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    import mlir.ir as ir
    import torch.fx


@dataclass
class BuildContext:
    """State shared by the MLIR builder and its lowering helpers."""

    host_function: HostFunction
    config: Config | object
    env: CompileEnvironment

    node_to_value: dict[torch.fx.Node, ir.Value] = field(default_factory=dict)
    param_to_value: dict[str, ir.Value] = field(default_factory=dict)

    block_id_to_size: dict[int, int] = field(default_factory=dict)
    block_hint_to_id: dict[int, int] = field(default_factory=dict)
    block_symint_to_id: dict[int, int] = field(default_factory=dict)
    block_id_to_upper_bound: dict[int, int] = field(default_factory=dict)

    block_id_to_iv: dict[int, ir.Value] = field(default_factory=dict)
    forall_insert_slices: list[tuple] = field(default_factory=list)
    for_store_ctx_stack: list[dict[str, Any]] = field(default_factory=list)
    for_block_id_stack: list[int] = field(default_factory=list)

    mlir_module: ir.Module | None = None
    mlir_context: ir.Context | None = None
    node_to_aten_func: dict[int, tuple[str, list]] = field(default_factory=dict)
