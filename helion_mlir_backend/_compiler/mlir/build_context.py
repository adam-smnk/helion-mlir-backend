"""Shared mutable state for MLIR code generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from .support import block_id_from_key
from .support import block_id_from_symbol

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

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
    block_id_to_out_dim: dict[int, int] = field(default_factory=dict)

    block_id_to_iv: dict[int, ir.Value] = field(default_factory=dict)
    placeholder_dim_to_block_id: dict[tuple[int, int], int] = field(
        default_factory=dict
    )
    forall_insert_slices: list[tuple] = field(default_factory=list)
    for_store_ctx_stack: list[dict[str, Any]] = field(default_factory=list)
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
        import sympy
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
                            self.block_hint_to_id,
                            self.block_symint_to_id,
                            self.block_id_to_upper_bound,
                        )
                        if 0 <= dim < len(resolved):
                            shape.append(resolved[dim])
                            continue
                value = shape_node.meta.get("val")
                if isinstance(value, torch.SymInt):
                    block_id = block_id_from_symbol(str(value))
                    if block_id is None:
                        expression = getattr(getattr(value, "node", None), "expr", None)
                        if isinstance(expression, sympy.Symbol):
                            block_id = block_id_from_symbol(str(expression))
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

    def build_sym_to_block_id(self) -> dict[str, int]:
        """Build symbolic ``uN`` names from the active Helion block sizes."""
        return {f"u{block.block_id}": block.block_id for block in self.env.block_sizes}

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

    def infer_block_id_from_index(
        self, index_node: object, sym_to_block_id: dict[str, int]
    ) -> int | None:
        """Infer the block id represented by an index expression."""
        block_id, _ = self.infer_index_block_and_bias(index_node, sym_to_block_id)
        return block_id

    def infer_index_block_and_bias(
        self, index_node: object, sym_to_block_id: dict[str, int]
    ) -> tuple[int | None, int]:
        """Infer a block id and additive bias from a tile-index expression."""
        import torch.fx

        block_id = self.infer_block_id_from_index_symbolic(index_node, sym_to_block_id)
        if block_id is not None:
            return block_id, 0
        if isinstance(index_node, int):
            return None, index_node
        if not isinstance(index_node, torch.fx.Node):
            return None, 0

        target_name = str(index_node.target)
        target_short_name = getattr(index_node.target, "__name__", "")
        if "aten.add" not in target_name and target_short_name not in (
            "add.Tensor",
            "add.default",
        ):
            return None, 0
        if len(index_node.args) < 2:
            return None, 0

        return self._infer_add_index_block_and_bias(
            index_node.args[0], index_node.args[1], sym_to_block_id
        )

    def _infer_add_index_block_and_bias(
        self,
        left_index: object,
        right_index: object,
        sym_to_block_id: dict[str, int],
    ) -> tuple[int | None, int]:
        left_block, left_bias = self.infer_index_block_and_bias(
            left_index, sym_to_block_id
        )
        right_block, right_bias = self.infer_index_block_and_bias(
            right_index, sym_to_block_id
        )
        if left_block is not None and right_block is None:
            return left_block, left_bias + right_bias
        if right_block is not None and left_block is None:
            return right_block, right_bias + left_bias
        if left_block is not None or right_block is not None:
            return None, 0

        left_shape_block = self.infer_block_id_from_value_shape(left_index)
        right_shape_block = self.infer_block_id_from_value_shape(right_index)
        if left_shape_block is not None and right_shape_block is None:
            return left_shape_block, left_bias + right_bias
        if right_shape_block is not None and left_shape_block is None:
            return right_shape_block, right_bias + left_bias
        return None, 0

    def infer_block_id_from_value_shape(self, index_node: object) -> int | None:
        """Infer a block id from a one-dimensional index value extent."""
        import mlir.ir as ir
        import torch.fx

        if not isinstance(index_node, torch.fx.Node):
            return None
        value = self.get_value(index_node)
        if value is None:
            return None
        try:
            value_type = ir.RankedTensorType(value.type)
        except Exception:
            return None
        if value_type.rank != 1:
            return None
        extent = int(value_type.shape[0])
        candidates = [
            block_id
            for block_id in self.block_id_to_iv
            if self.block_id_to_upper_bound.get(block_id) == extent
            or self.block_id_to_size.get(block_id) == extent
        ]
        return candidates[0] if len(candidates) == 1 else None

    def infer_block_id_from_index_symbolic(
        self, index_node: object, sym_to_block_id: dict[str, int]
    ) -> int | None:
        """Infer a block id from symbolic FX metadata."""
        import sympy
        import torch
        import torch.fx

        if not isinstance(index_node, torch.fx.Node):
            return None
        value = index_node.meta.get("val")
        if isinstance(value, torch.SymInt):
            # Symbol origins are authoritative; the uN name heuristic below is a
            # fallback and mis-maps grid symbols, whose names are unrelated to
            # block ids.
            origin_info = self.symbol_info(value)
            if origin_info is not None:
                return origin_info[0]
            symbol = str(value)
            if symbol in sym_to_block_id:
                return sym_to_block_id[symbol]
            expression = getattr(getattr(value, "node", None), "expr", None)
            if (
                isinstance(expression, sympy.Symbol)
                and id(expression) in self.block_symint_to_id
            ):
                return self.block_symint_to_id[id(expression)]

        target_name = getattr(index_node.target, "__name__", "")
        if target_name == "_get_symnode" and index_node.args:
            return block_id_from_key(index_node.args[0])
        if target_name not in ("sym_size.int", "sym_size_int"):
            return None
        if len(index_node.args) < 2:
            return None
        tensor_node = index_node.args[0]
        if not isinstance(tensor_node, torch.fx.Node):
            return None
        tensor_value = tensor_node.meta.get("val")
        if not isinstance(tensor_value, torch.Tensor):
            return None
        dimension_index = int(index_node.args[1])
        # The node's own metadata may have been replaced by a concrete hint, so
        # recover the block recorded when the shape was still symbolic.
        recorded = self.placeholder_dim_to_block_id.get(
            (id(tensor_node), dimension_index)
        )
        if recorded is not None:
            return recorded
        dimension = tensor_value.shape[dimension_index]
        dimension_origin = self.symbol_info(dimension)
        if dimension_origin is not None:
            return dimension_origin[0]
        dimension_expression = getattr(getattr(dimension, "node", None), "expr", None)
        if isinstance(dimension_expression, sympy.Symbol):
            resolved = self.block_symint_to_id.get(id(dimension_expression))
            if resolved is not None:
                return resolved
        return sym_to_block_id.get(str(dimension))

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
    def push_store_ctx(self, store_context: dict[str, Any]) -> Generator[None]:
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
