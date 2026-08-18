"""Core MLIR module builder for Helion kernels.

Walks the :class:`~helion._compiler.device_ir.DeviceIR` produced by the
standard Helion compilation pipeline and emits an ``mlir.ir.Module`` using
Linalg-on-Tensors abstraction.

Mapping summary
---------------
Helion construct          → MLIR construct
──────────────────────────────────────────────────────────────────────────────
Outer ``hl.tile([m,n])``  → ``scf.forall`` (parallel) + ``shared_outs``
Inner ``hl.tile(k)``      → ``scf.for``   (sequential, carries accumulator)
``hl.zeros([bm,bn])``     → ``tensor.empty`` + ``linalg.fill``
``hl.load(t, idx)``       → ``tensor.extract_slice``
``hl.store(t, idx, v)``   → ``tensor.parallel_insert_slice`` (in forall term.)
``torch.addmm``           → ``linalg.matmul``
``aten.mm``               → ``linalg.matmul`` (zero-init outs)
Pointwise aten ops        → ``linalg.generic`` or ``arith.*``
``_host_tensor(name)``    → reference to the corresponding function argument
``_get_symnode(bs_N)``    → the concrete block-size integer constant
``_phi``                  → the result of the enclosing ``scf.for``
``_new_var``              → pass-through (same MLIR value)

Why this module manually lowers ATen ops instead of using torch-mlir
---------------------------------------------------------------------
The `torch-mlir` project (present in this workspace as source) provides
``FxImporter`` + a ``torch-backend-to-linalg-on-tensors-backend-pipeline``
that would in principle replace the hand-written ``_lower_*`` methods below.
There are three reasons it is not used today:

1. **Not built**: ``torch_mlir`` is only present as source; using it requires
   a full LLVM + MLIR + torch-mlir CMake build.  Until a pre-built wheel is
   available (e.g. from the EUDSL index already in ``pyproject.toml``), the
   Python package cannot be imported.

2. **Dialect mismatch**: ``FxImporter`` imports an FX graph into the *torch*
   MLIR dialect (``torch.aten.mm``, ``torch.aten.add.Tensor``, …), not into
   linalg directly.  Reaching linalg-on-tensors requires running the C++
   pipeline pass ``torch-backend-to-linalg-on-tensors-backend-pipeline``
   afterwards.  We currently write linalg ops directly, bypassing the
   ``torch`` dialect entirely.

3. **Helion-specific ops are unknown to FxImporter**: Helion's device IR
   graph contains non-ATen call_function nodes (``_host_tensor``,
   ``_for_loop``, ``_phi``, ``_new_var``, ``store``, ``load``).
   ``FxImporter.import_nodes()`` only accepts ``TorchOpOverload`` or
   ``HigherOrderOperator`` targets; the helion ops would raise
   ``NotImplementedError``.  The tiling structure they encode must be emitted
   as ``scf.forall`` / ``scf.for``, which torch-mlir's pipeline never
   generates.

Desired future architecture (two-phase)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 1 – keep as-is: emit the ``scf.forall`` / ``scf.for`` tile structure
from helion-specific nodes (``_for_loop``, ``_host_tensor``, etc.).

Phase 2 – replace ``_lower_*`` ATen methods: extract the pure-ATen subgraph
from each tile body, import it with ``FxImporter.import_stateless_graph()``,
run the linalg lowering pipeline, then inline the resulting ``func.func``
into the tile body.  This would give automatic coverage for all ops that
torch-mlir supports (hundreds vs the current ~15) with no manual
reimplementation.

Prerequisites before Phase 2 can land:
  * A pre-built ``torch_mlir`` wheel on the EUDSL index (or a local build).
  * A bridge layer that strips helion-specific nodes before handing the
    subgraph to ``FxImporter``, then re-emits the surrounding structure.
  * Update ``pyproject.toml`` to add ``torch-mlir`` as a dependency
    (analogous to ``mlir-python-bindings``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.fx

from .aten_bridge import AtenHelperTable
from .build_context import BuildContext
from .lowering import lower_load
from .lowering import lower_nested_for_loop
from .lowering import lower_store
from .support import ModuleBuilderError
from .support import NodeLoweringError
from .support import UnsupportedOperationError
from .support import ValueNotFoundError
from .support import block_id_from_key
from .support import safe_int_conversion
from .support import torch_tensor_to_mlir_type

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    import mlir.ir as ir

log = logging.getLogger(__name__)


class MLIRModuleBuilder:
    """Builds an ``mlir.ir.Module`` from a compiled :class:`HostFunction`.

    Parameters
    ----------
    host_function:
        Compiled HostFunction (has ``device_ir`` populated).
    config:
        Helion Config with concrete block sizes.
    env:
        Active CompileEnvironment.
    """

    def __init__(
        self,
        host_function: HostFunction,
        config: object,
        env: CompileEnvironment,
    ) -> None:
        self.hf = host_function
        self.config = config
        self.env = env
        self.context = BuildContext(host_function, config, env)
        self.context.lower_node_callback = self._lower_node
        self._helper_table: AtenHelperTable | None = None

    def build(self) -> ir.Module:
        """Build and return the generated MLIR module."""
        import mlir.ir as ir

        try:
            ctx = ir.Context()
            from mlir.dialects import arith as arith_d  # noqa: F401
            from mlir.dialects import func as func_d  # noqa: F401
            from mlir.dialects import linalg as linalg_d  # noqa: F401
            from mlir.dialects import scf as scf_d  # noqa: F401
            from mlir.dialects import tensor as tensor_d  # noqa: F401

            with ir.Location.unknown(ctx):
                module = ir.Module.create()
                self.context.mlir_module = module
                self.context.mlir_context = ctx
                self._helper_table = AtenHelperTable(module)
                with ir.InsertionPoint(module.body), self.hf:
                    self._resolve_block_sizes()
                    self._resolve_block_upper_bounds()
                    self._prebuild_aten_helpers(module)
                    self._build_function()
            return module
        except (
            ModuleBuilderError,
            NodeLoweringError,
            ValueNotFoundError,
            UnsupportedOperationError,
        ):
            raise
        except Exception as exc:
            raise ModuleBuilderError(
                "module_creation",
                reason=str(exc),
                recovery_hint="Check that kernel has static_shapes=True and all ops are in hl.tile() loops",
            ) from exc

    def _build_function(self) -> None:
        from mlir.dialects import func as func_d
        import mlir.ir as ir

        tensor_params = [
            (name, value)
            for name, value in self.hf.params.arguments.items()
            if isinstance(value, torch.Tensor)
        ]
        out_name, out_tensor = self._find_output_tensor(tensor_params)
        output_types = [torch_tensor_to_mlir_type(out_tensor)]
        if any(name == out_name for name, _ in tensor_params):
            input_params = [
                (name, value) for name, value in tensor_params if name != out_name
            ]
        else:
            input_params = [
                (name, value)
                for name, value in tensor_params
                if value is not out_tensor
            ]
        input_types = [torch_tensor_to_mlir_type(value) for _, value in input_params]

        fn = func_d.FuncOp(self.hf.name, ir.FunctionType.get(input_types, output_types))
        fn.attributes["sym_visibility"] = ir.StringAttr.get("public")
        entry = fn.add_entry_block()
        with ir.InsertionPoint(entry):
            for (name, _), arg in zip(input_params, entry.arguments, strict=True):
                self.context.param_to_value[name] = arg
            if not self.context.block_id_to_size:
                with self.hf:
                    self._resolve_block_sizes()
            func_d.ReturnOp([self._build_kernel_body(out_tensor)])

    def _find_output_tensor(
        self, tensor_params: list[tuple[str, torch.Tensor]]
    ) -> tuple[str, torch.Tensor]:
        from .output_resolver import OutputTensorResolver

        return OutputTensorResolver(self.hf).resolve(tensor_params)

    def _resolve_block_sizes(self) -> None:
        """Populate ``_block_id_to_size`` and SymInt identity maps from the config."""
        for bs in self.env.block_sizes:
            # from_config requires HostFunction.current() — active via `with self.hf:`.
            config_val = bs.from_config(self.config)
            size = config_val if config_val is not None else bs.size
            if isinstance(size, torch.SymInt):
                try:
                    size = int(size)
                except Exception:
                    log.warning("block_id %d has dynamic size %s", bs.block_id, size)
                    size = -1
            else:
                size = int(size)
            self.context.block_id_to_size[bs.block_id] = size

        # Build hint-value → block_id by scanning _get_symnode nodes.
        import contextlib

        for graph_info in self.hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                tname = getattr(node.target, "__name__", "")
                if tname == "_get_symnode" and node.args:
                    key = node.args[0]
                    block_id = block_id_from_key(key)
                    if block_id is not None:
                        val = node.meta.get("val")
                        if (
                            val is not None
                            and block_id in self.context.block_id_to_size
                        ):
                            # Use sympy Symbol identity — stable because sympy caches symbols.
                            if isinstance(val, torch.SymInt):
                                import sympy as _sympy

                                expr = getattr(getattr(val, "node", None), "expr", None)
                                if isinstance(expr, _sympy.Symbol):
                                    self.context.block_symint_to_id[id(expr)] = block_id
                            with contextlib.suppress(TypeError, ValueError):
                                self.context.block_hint_to_id[int(val)] = block_id

    def _resolve_block_upper_bounds(self) -> None:
        """Infer static upper bounds per block_id from ``_for_loop`` nodes."""
        for graph_info in self.hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                if getattr(node.target, "__name__", "") != "_for_loop":
                    continue

                block_ids = node.args[1] if len(node.args) > 1 else None
                upper_bounds = node.args[2] if len(node.args) > 2 else None
                if not isinstance(block_ids, (list, tuple)):
                    continue
                if not isinstance(upper_bounds, (list, tuple)):
                    continue

                for bid, ub in zip(block_ids, upper_bounds, strict=False):
                    try:
                        block_id = int(bid)
                        ub_int = int(ub)
                    except Exception:
                        continue
                    if ub_int <= 0:
                        continue
                    prev = self.context.block_id_to_upper_bound.get(block_id)
                    if prev is None:
                        self.context.block_id_to_upper_bound[block_id] = ub_int
                    else:
                        self.context.block_id_to_upper_bound[block_id] = min(
                            prev, ub_int
                        )

    # ------------------------------------------------------------------
    # Kernel body – outer forall structure
    # ------------------------------------------------------------------

    def _build_kernel_body(self, out_tensor: torch.Tensor) -> ir.Value:
        from .lowering import build_kernel_body

        return build_kernel_body(self.context, out_tensor)

    # ------------------------------------------------------------------
    # Root graph processing
    # ------------------------------------------------------------------

    def _process_root_graphs(self, shared_out: ir.Value) -> ir.Value:
        return self.context.lower_root_graphs(shared_out)

    # ------------------------------------------------------------------
    # Graph walker
    # ------------------------------------------------------------------

    def _process_graph(self, graph: torch.fx.Graph) -> ir.Value | None:
        return self.context.lower_graph(graph)

    # ------------------------------------------------------------------
    # Per-node lowering dispatcher
    # ------------------------------------------------------------------

    def _lower_node(self, node: torch.fx.Node) -> ir.Value | None:
        """Dispatch a single FX node to the appropriate MLIR builder."""
        if node.op == "placeholder":
            # Handled when the graph is entered (for_loop iter args).
            return self.context.node_to_value.get(node)

        if node.op == "output":
            return None

        if node.op == "call_method":
            return self._lower_call_method(node)

        if node.op == "call_function":
            target = node.target
            tname = getattr(target, "__name__", str(target))

            from .support import lower_helion_node

            handled, value = lower_helion_node(self, node, tname)
            if handled:
                return value

            from .aten_bridge import lower_custom_aten

            lowered_custom = lower_custom_aten(self, node)
            if lowered_custom is not None:
                return lowered_custom
            if tname == "subscript":
                lowered_subscript = self._lower_subscript(node)
                if lowered_subscript is not None:
                    return lowered_subscript
                raise UnsupportedOperationError(
                    tname,
                    reason=f"Unsupported subscript form with args={node.args!r}",
                )
            # --- All standard ATen ops → pre-built linalg helper (func.call) ---
            from mlir.dialects import func as func_d

            from .aten_lowering import collect_tensor_input_positions
            from .aten_lowering import is_aten_op
            from .aten_lowering import normalized_aten_args

            if is_aten_op(node):
                entry = (
                    self._helper_table.get(id(node))
                    if self._helper_table is not None
                    else self.context.node_to_aten_func.get(id(node))
                )
                if entry is None:
                    log.warning(
                        "ATen node '%s' not found in pre-built helper map; "
                        "it may have failed during preprocessing.",
                        node.name,
                    )
                    raise UnsupportedOperationError(
                        tname,
                        reason="ATen node was not pre-lowered (check preprocessing warnings)",
                    )
                func_name, return_types = entry
                norm_args = normalized_aten_args(node)
                tensor_positions = collect_tensor_input_positions(node)
                input_mlir_vals = [
                    self._get_value(norm_args[i])
                    for i in tensor_positions
                    if isinstance(norm_args[i], torch.fx.Node)
                    and self._get_value(norm_args[i]) is not None
                ]

                if not self._helper_signature_matches(func_name, input_mlir_vals):
                    rebuilt = self._rebuild_aten_helper_for_call(
                        node,
                        input_mlir_vals,
                    )
                    if rebuilt is not None:
                        func_name, return_types = rebuilt

                if not self._helper_signature_matches(func_name, input_mlir_vals):
                    if len(input_mlir_vals) == 1 and self._helper_is_identity(
                        func_name
                    ):
                        return input_mlir_vals[0]
                    if len(input_mlir_vals) == 1 and len(return_types) == 1:
                        from .aten_bridge import lower_max_reduce_from_tensor

                        reduced = lower_max_reduce_from_tensor(
                            self.context, input_mlir_vals[0]
                        )
                        if reduced is not None and str(reduced.type) == str(
                            return_types[0]
                        ):
                            return reduced

                call = func_d.CallOp(return_types, func_name, input_mlir_vals)
                return call.results[0] if call.results else None

            # Not a helion op, not an ATen op.
            log.warning("Unhandled FX op: %s (target=%s)", node.name, tname)
            raise UnsupportedOperationError(
                tname,
                reason="Not a helion-specific op or a recognised ATen op",
            )

        return None

    def _lower_aten_matmul(self, node: torch.fx.Node) -> ir.Value | None:
        from .lowering import lower_matmul

        return lower_matmul(self.context, node)

    def _lower_aten_baddbmm(self, node: torch.fx.Node) -> ir.Value | None:
        from .lowering import lower_baddbmm

        return lower_baddbmm(self.context, node)

    def _lower_call_method(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower selected Tensor call-method ops.

        Supported today:
        - ``contiguous`` / ``clone`` / ``detach``: treated as aliases.
        - ``view`` / ``reshape`` only when shape is unchanged.

        Shape-changing view/reshape/flatten forms are not lowered yet in this
        path and must go through dedicated support.
        """
        method = str(node.target)

        if not node.args:
            return None

        base = node.args[0]
        base_val = self._get_value(base)
        if base_val is None:
            return None

        if method in ("contiguous", "clone", "detach"):
            return base_val

        if method in ("view", "reshape"):
            base_shape = self._shape_from_node_meta(base)
            result_shape = self._shape_from_node_meta(node)
            if (
                base_shape is not None
                and result_shape is not None
                and base_shape == result_shape
            ):
                return base_val
            raise UnsupportedOperationError(
                method,
                reason="Shape-changing call_method view/reshape lowering not implemented yet",
            )

        if method == "flatten":
            raise UnsupportedOperationError(
                method,
                reason="call_method flatten lowering not implemented yet",
            )

        return None

    def _shape_from_node_meta(self, node: object) -> list[int] | None:
        return self.context.shape_from_node_meta(node)

    # ------------------------------------------------------------------
    # Helpers for retrieving values
    # ------------------------------------------------------------------

    def _get_value(self, node_or_val: object) -> ir.Value | None:
        return self.context.get_value(node_or_val)

    # ------------------------------------------------------------------
    # Individual node lowering methods
    # ------------------------------------------------------------------

    def _lower_host_tensor(self, node: torch.fx.Node) -> ir.Value | None:
        from .lowering import lower_host_tensor

        return lower_host_tensor(self.context, node)

    def _lower_get_symnode(self, node: torch.fx.Node) -> ir.Value:
        """``_get_symnode('block_size_N')`` → integer index constant."""
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        key: str = node.args[0]
        # Parse "block_size_N" to get block_id N.
        block_id = block_id_from_key(key)
        if block_id is None:
            raise ValueNotFoundError(node, context=f"invalid block key: {key!r}")
        size = self.context.block_id_to_size.get(block_id, 0)
        idx = ir.IndexType.get()
        return arith_d.ConstantOp(idx, ir.IntegerAttr.get(idx, size)).result

    def _lower_tile_index(self, node: torch.fx.Node) -> ir.Value | None:
        from .lowering import lower_tile_index

        return lower_tile_index(self.context, node)

    def _lower_mask_to(self, node: torch.fx.Node) -> ir.Value | None:
        """Conservatively forward masked tensors when the backend has no mask IR."""
        if not node.args:
            return None
        return self._get_value(node.args[0])

    def _lower_subscript(self, node: torch.fx.Node) -> ir.Value | None:
        from .lowering import lower_subscript

        return lower_subscript(self.context, node)

    def _lower_sym_size(self, node: torch.fx.Node) -> ir.Value:
        """``sym_size.int(tensor, dim)`` → constant for the tensor dimension.

        Tolerates dynamic shapes (SymInt) by attempting resolution.
        """
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        # For static shapes the dimension is a concrete integer.
        val = node.meta.get("val")
        if isinstance(val, torch.SymInt):
            # Try to resolve SymInt
            try:
                concrete = int(val)
            except Exception:
                log.warning("Could not resolve SymInt in sym_size: %s", val)
                concrete = 0
        else:
            try:
                concrete = safe_int_conversion(val, "shape_dimension")
            except TypeError:
                log.warning("Could not convert shape dimension: %s", val)
                concrete = 0

        idx = ir.IndexType.get()
        return arith_d.ConstantOp(idx, ir.IntegerAttr.get(idx, concrete)).result

    def _lower_full(self, node: torch.fx.Node) -> ir.Value:
        from .lowering import lower_full

        return lower_full(self.context, node)

    def _lower_zeros(self, node: torch.fx.Node) -> ir.Value:
        from .lowering import lower_zeros

        return lower_zeros(self.context, node)

    def _lower_for_loop(self, node: torch.fx.Node) -> ir.Value:
        return lower_nested_for_loop(self.context, node)

    def _lower_load(self, node: torch.fx.Node) -> ir.Value:
        return lower_load(self.context, node)

    def _lower_store(self, node: torch.fx.Node) -> None:
        lower_store(self.context, node)

    def _lower_store_node(self, node: torch.fx.Node) -> ir.Value | None:
        self._lower_store(node)
        return None

    # ------------------------------------------------------------------
    # ATen pre-pass: lower all ATen nodes before codegen starts
    # ------------------------------------------------------------------

    def _prebuild_aten_helpers(self, module: ir.Module) -> None:
        """Scan device IR for ATen nodes, lower them all via one torch-mlir pass.

        Results are stored in the context's ATen helper map and the helper
        ``func.func`` operations are inserted at the module's top level.
        """
        from .aten_bridge import is_custom_aten
        from .aten_lowering import is_aten_op
        from .aten_lowering import preprocess_aten_nodes

        # Propagate symbolic shapes from outer _for_loop iter-args into inner
        # loop body placeholders BEFORE scanning ATen nodes.  Without this,
        # placeholder shapes in nested loop bodies are evaluated to their hint
        # values (e.g. both tile_m and tile_n evaluate to 64), making them
        # indistinguishable when resolving block_ids in _resolve_shape.
        from .support import restore_symbolic_shapes_in_bodies

        restore_symbolic_shapes_in_bodies(self.hf, self.context)
        self._refresh_aten_tensor_meta()

        aten_nodes: list[torch.fx.Node] = []
        for graph_info in self.hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if is_aten_op(node):
                    if is_custom_aten(node):
                        continue
                    aten_nodes.append(node)

        if not aten_nodes:
            return

        entries = preprocess_aten_nodes(
            aten_nodes,
            module,
            self.context.block_id_to_size,
            self.context.block_hint_to_id,
            self.context.block_symint_to_id,
            self.context.block_id_to_upper_bound,
        )
        self.context.node_to_aten_func = entries
        if self._helper_table is not None:
            self._helper_table.replace(entries)

    def _helper_signature_matches(
        self,
        func_name: str,
        input_mlir_vals: list[ir.Value],
    ) -> bool:
        """Return True when helper function arg types match provided MLIR values."""
        if self._helper_table is None:
            return False
        return self._helper_table.signature_matches(func_name, input_mlir_vals)

    def _helper_is_identity(self, func_name: str) -> bool:
        """Return True if helper body is `return arg0` with no intermediate ops."""
        if self._helper_table is None:
            return False
        return self._helper_table.is_identity(func_name)

    def _rebuild_aten_helper_for_call(
        self,
        node: torch.fx.Node,
        input_mlir_vals: list[ir.Value],
    ) -> tuple[str, list[ir.Type]] | None:
        from .aten_bridge import rebuild_aten_helper_for_call

        return rebuild_aten_helper_for_call(self.context, node, input_mlir_vals)

    def _refresh_aten_tensor_meta(self) -> None:
        from .support import refresh_aten_tensor_meta

        refresh_aten_tensor_meta(self.hf)
