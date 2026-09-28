"""Shared mutable state for MLIR code generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import arith as arith_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import sympy
import torch
import torch.fx
from torch.utils._sympy import functions as sympy_functions

from .lowering.tensor_state import TensorState
from .support import DynamicShapeError
from .support import block_id_from_key
from .support.index_meta import resolve_index_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config

    from .analysis.contractions import ContractionPlan
    from .analysis.geometry import KernelGeometry
    from .analysis.signature import KernelSignature
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
    # Elements of the current tile inside its loop, for loops whose last tile is
    # partial (Helion's tile mask); absent when every tile is full.
    block_id_to_valid: dict[int, int | ir.Value] = field(default_factory=dict)
    # Normalized (unit-step) trip index of an outer forall dimension, when known.
    block_id_to_trip_iv: dict[int, ir.Value] = field(default_factory=dict)
    tensors: TensorState = field(default_factory=TensorState)
    effects: TensorEffects | None = None
    signature: KernelSignature | None = None
    # Runtime scalar values of the current function, by ``_get_symnode`` key.
    scalars: dict[str, ir.Value] = field(default_factory=dict)

    mlir_module: ir.Module | None = None
    mlir_context: ir.Context | None = None
    aten_helpers: AtenHelperTable | None = None
    contractions: ContractionPlan | None = None
    lower_node_callback: Callable[[torch.fx.Node], ir.Value | None] | None = None

    # Size values of the current function by expression (see ``size``).
    size_values: dict[sympy.Expr, ir.Value] = field(default_factory=dict)
    _size_block: ir.Block | None = None
    _size_setup: int = 0
    _size_anchor: ir.Operation | None = None
    _block_symbols: dict | None = None
    _ref_sizes: dict[str, list[sympy.Expr]] = field(default_factory=dict)

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

    def shape_from_nodes(
        self, shape_nodes: list, operation_name: str = "op"
    ) -> list[int | ir.Value]:
        """Dims of a shape list: ints, tile sizes, constants or runtime sizes."""
        import torch.fx

        shape: list[int | ir.Value] = []
        for shape_node in shape_nodes:
            if isinstance(shape_node, int):
                shape.append(shape_node)
                continue
            if isinstance(shape_node, torch.SymInt):
                shape.append(self.size(shape_node))
                continue
            if isinstance(shape_node, torch.fx.Node):
                if shape_node.target is tracing_ops._get_symnode:
                    block_id = block_id_from_key(shape_node.args[0])
                    if block_id is not None and block_id in self.geometry.blocks:
                        shape.append(self.tile_size(block_id))
                        continue
                value = self.get_value(shape_node)
                owner = value.owner if value is not None else None
                if isinstance(owner, ir.OpView) and owner.name == "arith.constant":
                    shape.append(ir.IntegerAttr(owner.attributes["value"]).value)
                    continue
                meta = shape_node.meta.get("val")
                if isinstance(meta, torch.SymInt):
                    shape.append(self.size(meta))
                    continue
            raise DynamicShapeError(
                shape_nodes, symbol_name=f"{shape_node} in {operation_name}"
            )
        return shape

    def tile_size(self, block_id: int) -> int | ir.Value:
        """The static tile extent of ``block_id``, or the runtime span of a whole tile."""
        block = self.geometry.block(block_id)
        return self.size(block.span_expr) if block.whole else block.tile_extent

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    def begin_function(self, block: ir.Block) -> None:
        """Size values of the function with entry ``block`` go after its current ops."""
        self.size_values.clear()
        self._size_block = block
        self._size_setup = len(block.operations)
        self._size_anchor = None

    def size_expr(self, value: object) -> sympy.Expr:
        """``value`` (int, ``SymInt`` or expression) over size symbols only: the
        shape env's replacements applied and block sizes substituted."""
        if isinstance(value, torch.SymInt):
            value = value.node.expr
        expr = self.env.shape_env.replace(sympy.sympify(value))
        return expr.xreplace(self._block_size_symbols())

    def size(self, value: object) -> int | ir.Value:
        """A size as a static int or an ``index`` value, never its example value.

        A size symbol is ``tensor.dim`` of the first host tensor argument with
        that size, else the runtime scalar that carries it; compound expressions
        are ``index`` arithmetic. Values are emitted at the start of the current
        function (so they dominate every use) and shared by equal expressions.
        """
        expr = self.size_expr(value)
        if not expr.free_symbols:
            return int(expr)
        return self._size_value(expr)

    def ref_sizes(self, name: str) -> list[sympy.Expr]:
        """The size expressions of host tensor ``name``'s dims."""
        if name not in self._ref_sizes:
            fake = self.signature.refs[name].fake
            self._ref_sizes[name] = [self.size_expr(size) for size in fake.shape]
        return self._ref_sizes[name]

    def extent(
        self, value: ir.Value, dim: int, name: str | None = None
    ) -> int | ir.Value:
        """Size of ``value``'s dimension ``dim``. For the full extent of host tensor
        ``name`` it comes from :meth:`size`, so equal sizes are the same value."""
        size = ir.RankedTensorType(value.type).shape[dim]
        if not ir.ShapedType.is_dynamic_size(size):
            return size
        if name is not None and name in self.signature.refs:
            return self.size(self.ref_sizes(name)[dim])
        return tensor_d.DimOp(value, self.index_const(dim)).result

    def _block_size_symbols(self) -> dict:
        if self._block_symbols is None:
            replace = self.env.shape_env.replace
            self._block_symbols = {
                info.var.node.expr: replace(block.span_expr)
                if block.whole
                else sympy.Integer(block.block_size)
                for info in self.env.block_sizes
                if (block := self.geometry.blocks.get(info.block_id)) is not None
                and isinstance(info.var, torch.SymInt)
            }
        return self._block_symbols

    def _size_value(self, expr: sympy.Expr) -> ir.Value:
        value = self.size_values.get(expr)
        if value is None:
            with self._size_insertion_point():
                value = self._emit_size(expr)
            self.size_values[expr] = value
            if self._size_anchor is None:
                self._size_setup = len(self._size_block.operations)
        return value

    def _size_insertion_point(self) -> ir.InsertionPoint:
        if self._size_anchor is None:
            operations = list(self._size_block.operations)
            if len(operations) == self._size_setup:
                return ir.InsertionPoint(self._size_block)
            self._size_anchor = operations[self._size_setup]
        return ir.InsertionPoint(self._size_anchor)

    def _emit_size(self, expr: sympy.Expr) -> ir.Value:
        source = self._size_source(expr)
        if source is not None:
            return source
        operands = [
            self.index_const(int(arg)) if arg.is_Integer else self._size_value(arg)
            for arg in expr.args
        ]
        combine = None
        if isinstance(expr, sympy.Add):
            combine = arith_d.addi
        elif isinstance(expr, sympy.Mul):
            combine = arith_d.muli
        elif isinstance(expr, (sympy.Max, sympy_functions.Max)):
            combine = arith_d.maxsi
        elif isinstance(expr, (sympy.Min, sympy_functions.Min)):
            combine = arith_d.minsi
        elif isinstance(expr, sympy_functions.FloorDiv):
            combine = arith_d.floordivsi
        elif isinstance(expr, (sympy_functions.Mod, sympy_functions.PythonMod)):
            combine = arith_d.remsi
        if combine is None or not operands:
            raise DynamicShapeError(expr, symbol_name=self._describe(expr))
        return reduce(combine, operands)

    def _size_source(self, expr: sympy.Expr) -> ir.Value | None:
        for name in self.signature.refs:
            value = self.param_to_value.get(name)
            if value is None:
                continue
            for dim, size in enumerate(self.ref_sizes(name)):
                if size == expr:
                    return tensor_d.DimOp(value, self.index_const(dim)).result
        for key, scalar in self.signature.scalars.items():
            if (
                key in self.scalars
                and scalar.expr is not None
                and self.size_expr(scalar.expr) == expr
            ):
                return self.cast_to_index(self.scalars[key])
        return None

    def _describe(self, expr: sympy.Expr) -> str:
        origins = {
            str(symbol): origin.origin
            for symbol in expr.free_symbols
            if (origin := self.host_function.expr_to_origin.get(symbol)) is not None
        }
        return (
            f"size {expr} (origins {origins}), which no host tensor argument or "
            "runtime scalar of this function provides"
        )

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
        self.block_id_to_valid.clear()
        self.block_id_to_trip_iv.clear()
        self.tensors.clear()
        self.scalars.clear()
        self.size_values.clear()

    def bind_loop(
        self,
        block_id: int,
        offset: ir.Value,
        bounds: tuple[int | ir.Value, int | ir.Value],
    ) -> None:
        """Make ``offset`` the current tile of ``block_id`` in a loop over ``bounds``."""
        self.block_id_to_iv[block_id] = offset
        self.block_id_to_bounds[block_id] = bounds
        valid = self._valid_size(block_id, offset, *bounds)
        if valid is None:
            self.block_id_to_valid.pop(block_id, None)
        else:
            self.block_id_to_valid[block_id] = valid

    def _valid_size(
        self,
        block_id: int,
        offset: ir.Value,
        begin: int | ir.Value,
        end: int | ir.Value,
    ) -> int | ir.Value | None:
        """``min(tile, end - offset)``, or ``None`` if every tile is full."""
        from .lowering import emit

        if self.geometry.is_grid(block_id):
            return None
        size = self.geometry.tile_extent(block_id)
        d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
        if isinstance(begin, int) and isinstance(end, int):
            if (end - begin) % size == 0:
                return None
            if end - begin < size:
                return end - begin
            return emit.affine_min(
                [ir.AffineConstantExpr.get(size), ir.AffineConstantExpr.get(end) - d0],
                [offset],
            )
        return emit.affine_min(
            [ir.AffineConstantExpr.get(size), d1 - d0], [offset, self.as_index(end)]
        )

    @contextmanager
    def enter_for_loop(
        self,
        block_id: int,
        induction_variable: ir.Value,
        bounds: tuple[int | ir.Value, int | ir.Value],
    ) -> Generator[None]:
        """Bind a loop induction variable and bounds, restoring the previous ones."""
        saved = [
            (table, table.get(block_id))
            for table in (
                self.block_id_to_iv,
                self.block_id_to_bounds,
                self.block_id_to_valid,
            )
        ]
        self.bind_loop(block_id, induction_variable, bounds)
        try:
            yield
        finally:
            for table, previous in saved:
                if previous is None:
                    table.pop(block_id, None)
                else:
                    table[block_id] = previous
