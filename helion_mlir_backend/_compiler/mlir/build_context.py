"""Shared mutable state for MLIR code generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from .support import block_id_from_key
from .support.index_meta import resolve_index_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    import mlir.ir as ir
    import torch.fx

    from .lowering.for_store_context import ForStoreContext


@dataclass
class BuildContext:
    """State shared by the MLIR builder and its lowering helpers."""

    host_function: HostFunction
    config: Config | object
    env: CompileEnvironment

    node_to_value: dict[torch.fx.Node, ir.Value] = field(default_factory=dict)
    param_to_value: dict[str, ir.Value] = field(default_factory=dict)

    block_id_to_size: dict[int, int] = field(default_factory=dict)
    block_id_to_upper_bound: dict[int, int] = field(default_factory=dict)

    block_id_to_iv: dict[int, ir.Value] = field(default_factory=dict)
    forall_insert_slices: list[tuple] = field(default_factory=list)
    for_store_ctx_stack: list[ForStoreContext] = field(default_factory=list)
    for_block_id_stack: list[int] = field(default_factory=list)

    mlir_module: ir.Module | None = None
    mlir_context: ir.Context | None = None
    node_to_aten_func: dict[int, tuple[str, list]] = field(default_factory=dict)
    lower_node_callback: Callable[[torch.fx.Node], ir.Value | None] | None = None

    def get_value(self, node_or_value: object) -> ir.Value | None:
        """Look up an MLIR value for an FX node or scalar literal."""
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir
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
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        index_type = ir.IndexType.get()
        return arith_d.ConstantOp(
            index_type,
            ir.IntegerAttr.get(index_type, value),
        ).result

    def cast_to_index(self, value: ir.Value) -> ir.Value:
        """Cast an integer or rank-zero tensor value to MLIR index type."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        if isinstance(value.type, ir.IndexType):
            return value
        if isinstance(value.type, ir.RankedTensorType) and value.type.rank == 0:
            element_type = value.type.element_type
            if isinstance(element_type, (ir.IntegerType, ir.IndexType)):
                value = tensor_d.ExtractOp(value, []).result
                if isinstance(value.type, ir.IndexType):
                    return value
        return arith_d.IndexCastOp(ir.IndexType.get(), value).result

    def cast_scalar_to(self, value: ir.Value, target: ir.Type) -> ir.Value | None:
        """Convert a scalar MLIR value to the requested element type."""
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        source = value.type
        if str(source) == str(target):
            return value
        if isinstance(source, ir.IndexType):
            if isinstance(target, ir.IntegerType):
                return arith_d.IndexCastOp(target, value).result
            if isinstance(target, ir.FloatType):
                integer = arith_d.IndexCastOp(ir.IntegerType.get_signless(64), value)
                return arith_d.SIToFPOp(target, integer.result).result
            return None
        if isinstance(source, ir.IntegerType) and isinstance(target, ir.IndexType):
            return arith_d.IndexCastOp(target, value).result
        if isinstance(source, ir.IntegerType) and isinstance(target, ir.IntegerType):
            if source.width == target.width:
                return value
            operation = (
                arith_d.ExtSIOp if source.width < target.width else arith_d.TruncIOp
            )
            return operation(target, value).result
        if isinstance(source, ir.IntegerType) and isinstance(target, ir.FloatType):
            return arith_d.SIToFPOp(target, value).result
        if isinstance(source, ir.FloatType) and isinstance(target, ir.FloatType):
            if source.width == target.width:
                return value
            operation = (
                arith_d.ExtFOp if source.width < target.width else arith_d.TruncFOp
            )
            return operation(target, value).result
        if isinstance(source, ir.FloatType) and isinstance(target, ir.IntegerType):
            return arith_d.FPToSIOp(target, value).result
        return None

    def shape_from_node_meta(self, node: object) -> list[int] | None:
        """Extract a concrete tensor shape from FX metadata."""
        import torch.fx

        if not isinstance(node, torch.fx.Node):
            return None
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            try:
                return [int(dim) for dim in value.shape]
            except Exception:
                return None
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
        if shape is None:
            return None
        try:
            return [int(dim) for dim in shape]
        except Exception:
            return None

    def shape_from_nodes(
        self, shape_nodes: list, operation_name: str = "op"
    ) -> list[int]:
        """Resolve FX shape nodes using configured block sizes and metadata."""
        import torch.fx

        shape: list[int] = []
        for shape_node in shape_nodes:
            if isinstance(shape_node, torch.fx.Node):
                target_name = getattr(shape_node.target, "__name__", "")
                if target_name == "_get_symnode" and shape_node.args:
                    block_id = block_id_from_key(shape_node.args[0])
                    if block_id is not None and block_id in self.block_id_to_size:
                        shape.append(self.block_id_to_size[block_id])
                        continue
                if (
                    target_name in ("sym_size.int", "sym_size_int")
                    and len(shape_node.args) >= 2
                ):
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
                            self.block_id_to_size,
                            self.env,
                            self.block_id_to_upper_bound,
                        )
                        if 0 <= dim < len(resolved):
                            shape.append(resolved[dim])
                            continue
                value = shape_node.meta.get("val")
                if isinstance(value, torch.SymInt):
                    block_id = self.env.get_block_id(value)
                    if block_id is not None and block_id in self.block_id_to_size:
                        shape.append(self.block_id_to_size[block_id])
                        continue
                if value is not None:
                    try:
                        shape.append(int(value))
                        continue
                    except Exception:
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

    def infer_index_block_and_bias(self, index_node: object) -> tuple[int | None, int]:
        """Infer a block id and additive bias from a tile-index expression."""
        descriptor = resolve_index_descriptor(self, index_node)
        return descriptor.block_id, descriptor.bias

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

    def lower_root_graphs(self, shared_out: ir.Value) -> ir.Value:
        """Lower all root device graphs for the active forall body."""
        result = shared_out
        for root_id in self.host_function.device_ir.root_ids:
            graph = self.host_function.device_ir.graphs[root_id].graph
            result = self.lower_graph(graph) or result
        return result

    @contextmanager
    def enter_for_loop(
        self, block_id: int, induction_variable: ir.Value
    ) -> Generator[None]:
        """Bind a loop induction variable and restore the previous binding."""
        previous = self.block_id_to_iv.get(block_id)
        self.block_id_to_iv[block_id] = induction_variable
        self.for_block_id_stack.append(block_id)
        try:
            yield
        finally:
            if self.for_block_id_stack and self.for_block_id_stack[-1] == block_id:
                self.for_block_id_stack.pop()
            if previous is None:
                self.block_id_to_iv.pop(block_id, None)
            else:
                self.block_id_to_iv[block_id] = previous

    @contextmanager
    def push_store_ctx(self, store_context: ForStoreContext) -> Generator[None]:
        """Push and reliably remove a synthetic store context."""
        self.for_store_ctx_stack.append(store_context)
        try:
            yield
        finally:
            if (
                self.for_store_ctx_stack
                and self.for_store_ctx_stack[-1] is store_context
            ):
                self.for_store_ctx_stack.pop()
