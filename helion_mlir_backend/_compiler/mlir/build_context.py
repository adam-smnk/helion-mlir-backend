"""Shared mutable state for MLIR code generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import arith as arith_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from .lowering.tensor_state import TensorState
from .support import block_id_from_key
from .support.index_meta import resolve_index_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    import torch.fx

    from .analysis.contractions import ContractionPlan
    from .analysis.geometry import KernelGeometry
    from .analysis.tensor_effects import TensorEffects
    from .aten_bridge import AtenHelperTable


@dataclass
class BuildContext:
    """State shared by the MLIR builder and its lowering helpers."""

    host_function: HostFunction
    config: Config | object
    env: CompileEnvironment

    node_to_value: dict[torch.fx.Node, ir.Value] = field(default_factory=dict)
    param_to_value: dict[str, ir.Value] = field(default_factory=dict)

    geometry: KernelGeometry | None = None

    # Written directly once per outer grid block id (control_flow._bind_grid_iv) then
    # save/restored per nested scf.for level via enter_for_loop(); read
    # everywhere a block id's current induction variable is needed. The value is
    # the tile's absolute offset (``begin + trip * step``).
    block_id_to_iv: dict[int, ir.Value] = field(default_factory=dict)
    # The active loop's (begin, end) per block id, as static ints or index values.
    block_id_to_bounds: dict[int, tuple[int | ir.Value, int | ir.Value]] = field(
        default_factory=dict
    )
    # Normalized (unit-step) trip index of an outer forall dimension, when known.
    block_id_to_trip_iv: dict[int, ir.Value] = field(default_factory=dict)
    tensors: TensorState = field(default_factory=TensorState)
    effects: TensorEffects | None = None

    mlir_module: ir.Module | None = None
    mlir_context: ir.Context | None = None
    aten_helpers: AtenHelperTable | None = None
    contractions: ContractionPlan | None = None
    lower_node_callback: Callable[[torch.fx.Node], ir.Value | None] | None = None

    def get_value(self, node_or_value: object) -> ir.Value | None:
        """Look up an MLIR value for an FX node or scalar literal."""
        import torch.fx

        if isinstance(node_or_value, torch.fx.Node):
            return self.node_to_value.get(node_or_value)
        if isinstance(node_or_value, int):
            index_type = ir.IndexType.get()
            return arith_d.ConstantOp(
                index_type,
                ir.IntegerAttr.get(index_type, node_or_value),
            ).result
        if isinstance(node_or_value, float):
            float_type = ir.F32Type.get()
            return arith_d.ConstantOp(
                float_type,
                ir.FloatAttr.get(float_type, node_or_value),
            ).result
        return None

    def set_value(self, node: torch.fx.Node, value: ir.Value) -> None:
        """Associate an FX node with its generated MLIR value."""
        self.node_to_value[node] = value

    def index_const(self, value: int) -> ir.Value:
        """Create an MLIR index constant."""

        index_type = ir.IndexType.get()
        return arith_d.ConstantOp(
            index_type,
            ir.IntegerAttr.get(index_type, value),
        ).result

    def as_index(self, value: int | ir.Value) -> ir.Value:
        """An index value for a static int or an existing scalar value."""
        if isinstance(value, int):
            return self.index_const(value)
        return self.cast_to_index(value)

    def cast_to_index(self, value: ir.Value) -> ir.Value:
        """Cast an integer or rank-zero tensor value to MLIR index type."""

        if isinstance(value.type, ir.IndexType):
            return value
        if isinstance(value.type, ir.RankedTensorType) and value.type.rank == 0:
            element_type = value.type.element_type
            if isinstance(element_type, (ir.IntegerType, ir.IndexType)):
                value = tensor_d.ExtractOp(value, []).result
                if isinstance(value.type, ir.IndexType):
                    return value
        return arith_d.IndexCastOp(ir.IndexType.get(), value).result

    def shape_from_node_meta(self, node: object) -> list[int] | None:
        """Extract a concrete tensor shape from FX metadata."""
        import torch.fx

        if not isinstance(node, torch.fx.Node):
            return None
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            try:
                return [int(dim) for dim in value.shape]
            except (TypeError, ValueError):
                return None
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
        if shape is None:
            return None
        try:
            return [int(dim) for dim in shape]
        except (TypeError, ValueError):
            return None

    def shape_from_nodes(
        self, shape_nodes: list, operation_name: str = "op"
    ) -> list[int]:
        """Resolve FX shape nodes using configured block sizes and metadata."""
        import torch.fx

        shape: list[int] = []
        for shape_node in shape_nodes:
            if isinstance(shape_node, torch.fx.Node):
                target = shape_node.target
                if target is tracing_ops._get_symnode and shape_node.args:
                    block_id = block_id_from_key(shape_node.args[0])
                    if block_id is not None and block_id in self.geometry.blocks:
                        shape.append(self.geometry.tile_extent(block_id))
                        continue
                if target is torch.ops.aten.sym_size.int and len(shape_node.args) >= 2:
                    tensor_node, dim = shape_node.args[:2]
                    value = (
                        tensor_node.meta.get("val")
                        if isinstance(tensor_node, torch.fx.Node)
                        else None
                    )
                    if isinstance(value, torch.Tensor) and isinstance(dim, int):
                        from helion_mlir_backend._compiler.mlir.aten_lowering import (
                            _resolve_dims,
                        )

                        resolved = _resolve_dims(
                            value.shape,
                            self.geometry.block_sizes(),
                            self.env,
                            self.geometry.spans(),
                        )
                        if 0 <= dim < len(resolved):
                            shape.append(resolved[dim])
                            continue
                value = shape_node.meta.get("val")
                if isinstance(value, torch.SymInt):
                    block_id = self.env.get_block_id(value)
                    if block_id is not None and block_id in self.geometry.blocks:
                        shape.append(self.geometry.block_size(block_id))
                        continue
                if value is not None:
                    try:
                        shape.append(int(value))
                        continue
                    except (TypeError, ValueError):
                        pass
            elif isinstance(shape_node, int):
                shape.append(shape_node)
                continue
            shape.append(1)
        return shape

    def symbol_info(self, value: object) -> tuple[int, str] | None:
        """Resolve a SymInt to ``(block_id, kind)`` using Helion symbol origins."""
        from .support import symbol_origin_info

        return symbol_origin_info(self.host_function, value)

    def node_symbol_info(self, node: object) -> tuple[int, str] | None:
        """Resolve an FX node's SymInt metadata to ``(block_id, kind)``."""
        import torch
        import torch.fx

        if not isinstance(node, torch.fx.Node):
            return None
        value = node.meta.get("val")
        if not isinstance(value, torch.SymInt):
            return None
        return self.symbol_info(value)

    def is_scalar_index_node(self, node: object) -> bool:
        """Return whether an index node denotes a scalar position, not a tile."""
        from .support import SCALAR_SYMBOL_KINDS

        info = self.node_symbol_info(node)
        return info is not None and info[1] in SCALAR_SYMBOL_KINDS

    def has_symint_operand(self, node: object) -> bool:
        """Return whether any operand is a symbolic scalar.

        torch-mlir cannot import SymInt operands, so these nodes are lowered
        directly instead of through a generated helper.
        """
        import torch
        import torch.fx

        if not isinstance(node, torch.fx.Node):
            return False
        arguments = list(node.args) + list(node.kwargs.values())
        return any(
            isinstance(argument, torch.fx.Node)
            and isinstance(argument.meta.get("val"), torch.SymInt)
            for argument in arguments
        )

    def infer_block_id_from_index(self, index_node: object) -> int | None:
        """Infer the block id represented by an index expression."""
        return resolve_index_descriptor(self, index_node).block_id

    def lower_graph(self, graph: torch.fx.Graph) -> ir.Value | None:
        """Lower an FX graph through the builder callback."""
        if self.lower_node_callback is None:
            raise RuntimeError("BuildContext.lower_node_callback is not configured")
        last_value: ir.Value | None = None
        for node in graph.nodes:
            value = self.lower_node_callback(node)
            if value is not None:
                self.set_value(node, value)
                last_value = value
        return last_value

    def reset_for_new_function(self) -> None:
        """Clear per-function state before building another ``func.func``.

        Needed only when compiling more than one function against the same
        ``BuildContext`` (one per phase): a previous function's parameter
        bindings, induction variables, and tensor states are SSA values
        scoped to that function and must not leak into the next one.
        """
        self.param_to_value.clear()
        self.block_id_to_iv.clear()
        self.block_id_to_bounds.clear()
        self.block_id_to_trip_iv.clear()
        self.tensors.clear()

    @contextmanager
    def enter_for_loop(
        self,
        block_id: int,
        induction_variable: ir.Value,
        bounds: tuple[int | ir.Value, int | ir.Value],
    ) -> Generator[None]:
        """Bind a loop induction variable and bounds, restoring the previous ones."""
        previous = self.block_id_to_iv.get(block_id)
        previous_bounds = self.block_id_to_bounds.get(block_id)
        self.block_id_to_iv[block_id] = induction_variable
        self.block_id_to_bounds[block_id] = bounds
        try:
            yield
        finally:
            if previous is None:
                self.block_id_to_iv.pop(block_id, None)
            else:
                self.block_id_to_iv[block_id] = previous
            if previous_bounds is None:
                self.block_id_to_bounds.pop(block_id, None)
            else:
                self.block_id_to_bounds[block_id] = previous_bounds
