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
from typing import Any

import torch
import torch.fx

from .block_ids import block_id_from_key
from .build_context import BuildContext
from .errors import DynamicShapeError
from .errors import ModuleBuilderError
from .errors import NodeLoweringError
from .errors import ShapeError
from .errors import UnsupportedOperationError
from .errors import ValueNotFoundError
from .errors import safe_int_conversion
from .type_utils import get_zero_attr
from .type_utils import torch_dtype_to_mlir
from .type_utils import torch_tensor_to_mlir_type

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

    @property
    def _node_to_value(self) -> dict[torch.fx.Node, ir.Value]:
        return self.context.node_to_value

    @property
    def _param_to_value(self) -> dict[str, ir.Value]:
        return self.context.param_to_value

    @property
    def _block_id_to_size(self) -> dict[int, int]:
        return self.context.block_id_to_size

    @property
    def _block_hint_to_id(self) -> dict[int, int]:
        return self.context.block_hint_to_id

    @property
    def _block_symint_to_id(self) -> dict[int, int]:
        return self.context.block_symint_to_id

    @property
    def _block_id_to_upper_bound(self) -> dict[int, int]:
        return self.context.block_id_to_upper_bound

    @property
    def _block_id_to_iv(self) -> dict[int, ir.Value]:
        return self.context.block_id_to_iv

    @property
    def _forall_insert_slices(self) -> list[tuple]:
        return self.context.forall_insert_slices

    @property
    def _for_store_ctx_stack(self) -> list[dict[str, Any]]:
        return self.context.for_store_ctx_stack

    @property
    def _for_block_id_stack(self) -> list[int]:
        return self.context.for_block_id_stack

    @property
    def _mlir_module(self) -> ir.Module | None:
        return self.context.mlir_module

    @_mlir_module.setter
    def _mlir_module(self, value: ir.Module | None) -> None:
        self.context.mlir_module = value

    @property
    def _mlir_context(self) -> ir.Context | None:
        return self.context.mlir_context

    @_mlir_context.setter
    def _mlir_context(self, value: ir.Context | None) -> None:
        self.context.mlir_context = value

    @property
    def _node_to_aten_func(self) -> dict[int, tuple[str, list]]:
        return self.context.node_to_aten_func

    @_node_to_aten_func.setter
    def _node_to_aten_func(self, value: dict[int, tuple[str, list]]) -> None:
        self.context.node_to_aten_func = value

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
                self._mlir_module = module
                self._mlir_context = ctx
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
                self._param_to_value[name] = arg
            if not self._block_id_to_size:
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
            self._block_id_to_size[bs.block_id] = size

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
                        if val is not None and block_id in self._block_id_to_size:
                            # Use sympy Symbol identity — stable because sympy caches symbols.
                            if isinstance(val, torch.SymInt):
                                import sympy as _sympy

                                expr = getattr(getattr(val, "node", None), "expr", None)
                                if isinstance(expr, _sympy.Symbol):
                                    self._block_symint_to_id[id(expr)] = block_id
                            with contextlib.suppress(TypeError, ValueError):
                                self._block_hint_to_id[int(val)] = block_id

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
                    prev = self._block_id_to_upper_bound.get(block_id)
                    if prev is None:
                        self._block_id_to_upper_bound[block_id] = ub_int
                    else:
                        self._block_id_to_upper_bound[block_id] = min(prev, ub_int)

    # ------------------------------------------------------------------
    # Kernel body – outer forall structure
    # ------------------------------------------------------------------

    def _build_kernel_body(self, out_tensor: torch.Tensor) -> ir.Value:
        from .control_flow import build_kernel_body

        return build_kernel_body(self, out_tensor)

    # ------------------------------------------------------------------
    # Root graph processing
    # ------------------------------------------------------------------

    def _process_root_graphs(self, shared_out: ir.Value) -> ir.Value:
        """Walk all root graphs and return the accumulated result.

        For a single root graph (the typical case), this processes the body
        of one forall tile iteration.
        """
        device_ir = self.hf.device_ir
        result: ir.Value = shared_out

        for root_id in device_ir.root_ids:
            root_graph_info = device_ir.graphs[root_id]
            result = self._process_graph(root_graph_info.graph)

        return result

    # ------------------------------------------------------------------
    # Graph walker
    # ------------------------------------------------------------------

    def _process_graph(self, graph: torch.fx.Graph) -> ir.Value | None:
        """Walk an FX graph and emit MLIR ops for each node.

        Returns the "last meaningful value" produced by the graph (used as the
        result of ``scf.for`` iterations or the forall body).
        """
        last_val: ir.Value | None = None

        for node in graph.nodes:
            val = self._lower_node(node)
            if val is not None:
                self._node_to_value[node] = val
                last_val = val

        return last_val

    # ------------------------------------------------------------------
    # Per-node lowering dispatcher
    # ------------------------------------------------------------------

    def _lower_node(self, node: torch.fx.Node) -> ir.Value | None:
        """Dispatch a single FX node to the appropriate MLIR builder."""
        if node.op == "placeholder":
            # Handled when the graph is entered (for_loop iter args).
            return self._node_to_value.get(node)

        if node.op == "output":
            return None

        if node.op == "call_method":
            return self._lower_call_method(node)

        if node.op == "call_function":
            target = node.target
            tname = getattr(target, "__name__", str(target))

            from .node_dispatch import lower_helion_node

            handled, value = lower_helion_node(self, node, tname)
            if handled:
                return value

            # Special-case addmm in nested reductions: lowering directly to
            # linalg.matmul with the accumulator as the `outs` tensor preserves
            # loop-carried equivalence required by downstream lighthouse passes.
            target_name = str(node.target)
            is_addmm = "aten.addmm" in target_name or tname == "addmm.default"
            is_add_tensor = "aten.add.Tensor" in target_name or tname == "add.Tensor"
            is_mm_like = (
                "aten.mm" in target_name
                or "aten.matmul" in target_name
                or tname in ("mm.default", "matmul.default")
            )
            is_bmm_like = "aten.bmm" in target_name or tname == "bmm.default"
            is_baddbmm = "aten.baddbmm" in target_name or tname == "baddbmm.default"

            if is_addmm:
                lowered_addmm = self._lower_aten_addmm(node)
                if lowered_addmm is not None:
                    return lowered_addmm
            if is_baddbmm:
                lowered_baddbmm = self._lower_aten_baddbmm(node)
                if lowered_baddbmm is not None:
                    return lowered_baddbmm
            if is_add_tensor:
                lowered_add_matmul = self._lower_aten_add_matmul_accumulate(node)
                if lowered_add_matmul is not None:
                    return lowered_add_matmul
                lowered_add_tensor = self._lower_aten_add_tensor(node)
                if lowered_add_tensor is not None:
                    return lowered_add_tensor
            if is_mm_like or is_bmm_like:
                lowered_matmul = self._lower_aten_matmul(node)
                if lowered_matmul is not None:
                    return lowered_matmul
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
                passthrough = self._lower_aten_passthrough(node)
                if passthrough is not None:
                    return passthrough

                reduce_max = self._lower_aten_reduce_max_1d(node)
                if reduce_max is not None:
                    return reduce_max

                entry = self._node_to_aten_func.get(id(node))
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
                    else:
                        passthrough = self._lower_aten_passthrough(node)
                        if passthrough is not None:
                            return passthrough

                if not self._helper_signature_matches(func_name, input_mlir_vals):
                    if len(input_mlir_vals) == 1 and self._helper_is_identity(
                        func_name
                    ):
                        return input_mlir_vals[0]
                    if len(input_mlir_vals) == 1 and len(return_types) == 1:
                        reduced = self._lower_max_reduce_from_tensor(input_mlir_vals[0])
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

    def _lower_aten_addmm(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower ``aten.addmm`` directly to ``linalg.matmul`` when possible."""
        from mlir.dialects import linalg as linalg_d

        from .aten_lowering import normalized_aten_args

        args = list(normalized_aten_args(node))
        if len(args) < 3:
            return None

        acc = self._get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
        lhs = self._get_value(args[1]) if isinstance(args[1], torch.fx.Node) else None
        rhs = self._get_value(args[2]) if isinstance(args[2], torch.fx.Node) else None
        if acc is None or lhs is None or rhs is None:
            return None

        beta = args[3] if len(args) > 3 else 1
        alpha = args[4] if len(args) > 4 else 1
        if beta != 1 or alpha != 1:
            return None

        return linalg_d.matmul(lhs, rhs, outs=[acc])

    def _lower_aten_add_matmul_accumulate(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower ``acc + matmul(lhs, rhs)`` to ``linalg.matmul(..., outs=[acc])``.

        This keeps the reduction update anchored on the loop-carried accumulator,
        which is required by downstream scf.for iter-arg equivalence checks.
        """
        from .aten_lowering import normalized_aten_args

        args = list(normalized_aten_args(node))
        if len(args) < 2:
            return None

        alpha = args[2] if len(args) > 2 else 1
        if alpha != 1:
            return None

        acc_node: torch.fx.Node | None = None
        matmul_node: torch.fx.Node | None = None

        for first, second in ((args[0], args[1]), (args[1], args[0])):
            if not isinstance(first, torch.fx.Node) or not isinstance(
                second, torch.fx.Node
            ):
                continue
            second_name = str(second.target)
            second_tname = getattr(second.target, "__name__", "")
            if (
                "aten.mm" in second_name
                or "aten.matmul" in second_name
                or "aten.bmm" in second_name
                or second_tname in ("mm.default", "matmul.default", "bmm.default")
            ):
                acc_node = first
                matmul_node = second
                break

        if acc_node is None or matmul_node is None:
            return None

        acc = self._get_value(acc_node)
        if acc is None:
            return None

        mat_args = list(normalized_aten_args(matmul_node))
        if len(mat_args) < 2:
            return None

        lhs = (
            self._get_value(mat_args[0])
            if isinstance(mat_args[0], torch.fx.Node)
            else None
        )
        rhs = (
            self._get_value(mat_args[1])
            if isinstance(mat_args[1], torch.fx.Node)
            else None
        )
        if lhs is None or rhs is None:
            return None

        return self._emit_matmul_like(lhs, rhs, out=acc)

    def _lower_aten_add_tensor(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower elementwise ``aten.add.Tensor`` directly to ``linalg.generic``."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import linalg as linalg_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        from .aten_lowering import normalized_aten_args

        args = list(normalized_aten_args(node))
        if len(args) < 2:
            return None

        alpha = args[2] if len(args) > 2 else 1
        if alpha != 1:
            return None

        lhs = self._get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
        rhs = self._get_value(args[1]) if isinstance(args[1], torch.fx.Node) else None

        # Tensor + scalar (or scalar + tensor): materialize a filled tensor and add.
        if lhs is not None and rhs is None and isinstance(args[1], (int, float)):
            lhs_ty = ir.RankedTensorType(lhs.type)
            elem_ty = lhs_ty.element_type
            if isinstance(elem_ty, ir.FloatType):
                scalar_attr = ir.FloatAttr.get(elem_ty, float(args[1]))
            elif isinstance(elem_ty, ir.IntegerType):
                scalar_attr = ir.IntegerAttr.get(elem_ty, int(args[1]))
            else:
                return None

            scalar_val = arith_d.ConstantOp(elem_ty, scalar_attr).result
            rhs_empty = tensor_d.EmptyOp(list(lhs_ty.shape), elem_ty).result
            rhs_tensor = linalg_d.fill(scalar_val, outs=[rhs_empty])
            out = tensor_d.EmptyOp(list(lhs_ty.shape), elem_ty).result
            return linalg_d.add(lhs, rhs_tensor, outs=[out])

        if rhs is not None and lhs is None and isinstance(args[0], (int, float)):
            rhs_ty = ir.RankedTensorType(rhs.type)
            elem_ty = rhs_ty.element_type
            if isinstance(elem_ty, ir.FloatType):
                scalar_attr = ir.FloatAttr.get(elem_ty, float(args[0]))
            elif isinstance(elem_ty, ir.IntegerType):
                scalar_attr = ir.IntegerAttr.get(elem_ty, int(args[0]))
            else:
                return None

            scalar_val = arith_d.ConstantOp(elem_ty, scalar_attr).result
            lhs_empty = tensor_d.EmptyOp(list(rhs_ty.shape), elem_ty).result
            lhs_tensor = linalg_d.fill(scalar_val, outs=[lhs_empty])
            out = tensor_d.EmptyOp(list(rhs_ty.shape), elem_ty).result
            return linalg_d.add(lhs_tensor, rhs, outs=[out])

        if lhs is None or rhs is None:
            return None

        lhs_ty = ir.RankedTensorType(lhs.type)
        rhs_ty = ir.RankedTensorType(rhs.type)
        lhs_shape = [int(d) for d in lhs_ty.shape]
        rhs_shape = [int(d) for d in rhs_ty.shape]
        out_rank = max(len(lhs_shape), len(rhs_shape))
        out_shape_rev: list[int] = []
        for i in range(out_rank):
            ld = lhs_shape[-1 - i] if i < len(lhs_shape) else 1
            rd = rhs_shape[-1 - i] if i < len(rhs_shape) else 1
            if ld != rd and ld != 1 and rd != 1:
                return None
            out_shape_rev.append(max(ld, rd))
        out_shape = list(reversed(out_shape_rev))

        lhs_elem = lhs_ty.element_type
        rhs_elem = rhs_ty.element_type

        def _promoted_elem_type() -> ir.Type | None:
            if str(lhs_elem) == str(rhs_elem):
                return lhs_elem
            if isinstance(lhs_elem, ir.IntegerType) and isinstance(
                rhs_elem, ir.IntegerType
            ):
                return lhs_elem if lhs_elem.width >= rhs_elem.width else rhs_elem
            if isinstance(lhs_elem, ir.FloatType) and isinstance(
                rhs_elem, ir.FloatType
            ):
                return lhs_elem if lhs_elem.width >= rhs_elem.width else rhs_elem
            if isinstance(lhs_elem, ir.FloatType) and isinstance(
                rhs_elem, ir.IntegerType
            ):
                return lhs_elem
            if isinstance(lhs_elem, ir.IntegerType) and isinstance(
                rhs_elem, ir.FloatType
            ):
                return rhs_elem
            return None

        out_elem = _promoted_elem_type()
        if out_elem is None:
            return None

        def _cast_scalar(val: ir.Value, src: ir.Type, dst: ir.Type) -> ir.Value | None:
            if str(src) == str(dst):
                return val
            if isinstance(src, ir.IntegerType) and isinstance(dst, ir.IntegerType):
                if src.width < dst.width:
                    return arith_d.ExtSIOp(dst, val).result
                return arith_d.TruncIOp(dst, val).result
            if isinstance(src, ir.IntegerType) and isinstance(dst, ir.FloatType):
                return arith_d.SIToFPOp(dst, val).result
            if isinstance(src, ir.FloatType) and isinstance(dst, ir.FloatType):
                if src.width < dst.width:
                    return arith_d.ExtFOp(dst, val).result
                return arith_d.TruncFOp(dst, val).result
            return None

        # General broadcast-compatible tensor + tensor lowering.
        if list(lhs_ty.shape) != list(rhs_ty.shape) or str(lhs_elem) != str(rhs_elem):
            out_ty = ir.RankedTensorType.get(out_shape, out_elem)
            out = tensor_d.GenerateOp(out_ty, [])
            body = out.operation.regions[0].blocks.append(
                *([ir.IndexType.get()] * len(out_shape))
            )
            with ir.InsertionPoint(body):
                ivs = list(body.arguments)

                def _indices_for_operand(op_shape: list[int]) -> list[ir.Value]:
                    idxs: list[ir.Value] = []
                    rank_delta = len(out_shape) - len(op_shape)
                    for dim, size in enumerate(op_shape):
                        if size == 1:
                            idxs.append(self._get_index_const(0))
                        else:
                            idxs.append(ivs[rank_delta + dim])
                    return idxs

                lhs_val = tensor_d.ExtractOp(
                    lhs,
                    _indices_for_operand(lhs_shape),
                    results=[lhs_elem],
                ).result
                rhs_val = tensor_d.ExtractOp(
                    rhs,
                    _indices_for_operand(rhs_shape),
                    results=[rhs_elem],
                ).result

                lhs_cast = _cast_scalar(lhs_val, lhs_elem, out_elem)
                rhs_cast = _cast_scalar(rhs_val, rhs_elem, out_elem)
                if lhs_cast is None or rhs_cast is None:
                    return None

                if isinstance(out_elem, ir.FloatType):
                    summed = arith_d.AddFOp(lhs_cast, rhs_cast).result
                elif isinstance(out_elem, ir.IntegerType):
                    summed = arith_d.AddIOp(lhs_cast, rhs_cast).result
                else:
                    return None

                tensor_d.YieldOp(summed)

            return out.result

        if len(lhs_ty.shape) != len(rhs_ty.shape):
            return None

        out = tensor_d.EmptyOp(list(lhs_ty.shape), lhs_elem).result
        return linalg_d.add(lhs, rhs, outs=[out])

    def _emit_matmul_like(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        out: ir.Value | None = None,
    ) -> ir.Value | None:
        from .matmul_ops import emit_matmul_like

        return emit_matmul_like(self, lhs, rhs, out)

    def _lower_aten_matmul(self, node: torch.fx.Node) -> ir.Value | None:
        from .matmul_ops import lower_matmul

        return lower_matmul(self, node)

    def _lower_aten_baddbmm(self, node: torch.fx.Node) -> ir.Value | None:
        from .matmul_ops import lower_baddbmm

        return lower_baddbmm(self, node)

    def _lower_aten_relu(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower ``aten.relu`` directly to elementwise ``linalg.generic``."""
        from mlir.dialects import linalg as linalg_d
        from mlir.dialects import tensor as tensor_d

        from .aten_lowering import normalized_aten_args

        args = list(normalized_aten_args(node))
        if len(args) < 1:
            return None

        inp = self._get_value(args[0]) if isinstance(args[0], torch.fx.Node) else None
        if inp is None:
            return None

        import mlir.ir as ir

        inp_ty = ir.RankedTensorType(inp.type)
        elem_ty = inp_ty.element_type
        if not isinstance(elem_ty, ir.FloatType):
            return None

        out = tensor_d.EmptyOp(list(inp_ty.shape), elem_ty).result
        zero_empty = tensor_d.EmptyOp(list(inp_ty.shape), elem_ty).result
        zero_cst = ir.FloatAttr.get(elem_ty, 0.0)
        from mlir.dialects import arith as arith_d

        zero_val = arith_d.ConstantOp(elem_ty, zero_cst).result
        zero_tensor = linalg_d.fill(zero_val, outs=[zero_empty])

        return linalg_d.max(inp, zero_tensor, outs=[out])

    def _lower_aten_passthrough(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower shape-preserving unary ATen ops as direct pass-through."""
        from .aten_lowering import normalized_aten_args

        target_name = str(node.target)
        tname = getattr(node.target, "__name__", "")
        passthrough_ops = (
            "aten.alias",
            "aten.detach",
            "aten.clone",
            "aten.contiguous",
        )
        passthrough_overloads = {
            "alias.default",
            "detach.default",
            "clone.default",
            "contiguous.default",
        }

        if (
            not any(op in target_name for op in passthrough_ops)
            and tname not in passthrough_overloads
        ):
            return None

        args = list(normalized_aten_args(node))
        if not args:
            return None

        if isinstance(args[0], torch.fx.Node):
            return self._get_value(args[0])

        return None

    def _lower_aten_reduce_max_1d(self, node: torch.fx.Node) -> ir.Value | None:
        """Lower 1D integer `aten.max` reductions to a scalar tensor."""

        target_name = str(node.target)
        tname = getattr(node.target, "__name__", "")
        is_max = "aten.max" in target_name or tname == "max.default"
        if not is_max:
            return None

        from .aten_lowering import normalized_aten_args

        args = list(normalized_aten_args(node))
        if not args or not isinstance(args[0], torch.fx.Node):
            return None

        inp = self._get_value(args[0])
        if inp is None:
            return None

        return self._lower_max_reduce_from_tensor(inp)

    def _lower_max_reduce_from_tensor(self, inp: ir.Value) -> ir.Value | None:
        """Lower max reduction for a rank-1 integer tensor to rank-0 integer tensor."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import scf as scf_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        inp_ty = ir.RankedTensorType(inp.type)
        if inp_ty.rank != 1 or not isinstance(inp_ty.element_type, ir.IntegerType):
            return None

        elem_ty = inp_ty.element_type
        n = int(inp_ty.shape[0])

        if n <= 0:
            return None

        bitwidth = elem_ty.width
        min_val = -(1 << (bitwidth - 1))
        init = arith_d.ConstantOp(elem_ty, ir.IntegerAttr.get(elem_ty, min_val)).result

        lb = self._get_index_const(0)
        ub = self._get_index_const(n)
        step = self._get_index_const(1)

        loop = scf_d.ForOp(lb, ub, step, iter_args=[init])
        with ir.InsertionPoint(loop.body):
            iv = loop.body.arguments[0]
            acc = loop.body.arguments[1]
            cur = tensor_d.ExtractOp(inp, [iv], results=[elem_ty]).result
            nxt = arith_d.MaxSIOp(cur, acc).result
            scf_d.YieldOp([nxt])

        scalar_empty = tensor_d.EmptyOp([], elem_ty).result
        return tensor_d.InsertOp(loop.results[0], scalar_empty, []).result

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
        """Extract concrete shape list from an FX node's metadata when possible."""
        if not isinstance(node, torch.fx.Node):
            return None

        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            try:
                return [int(d) for d in val.shape]
            except Exception:
                return None

        tmeta = node.meta.get("tensor_meta")
        if tmeta is None:
            return None
        shape = getattr(tmeta, "shape", None)
        if shape is None:
            return None
        try:
            return [int(d) for d in shape]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers for retrieving values
    # ------------------------------------------------------------------

    def _get_value(self, node_or_val: object) -> ir.Value | None:
        """Look up an MLIR Value for a node or constant."""
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        if isinstance(node_or_val, torch.fx.Node):
            return self._node_to_value.get(node_or_val)
        if isinstance(node_or_val, (int, float)):
            # Scalar constant – create an index constant for now.
            if isinstance(node_or_val, int):
                idx = ir.IndexType.get()
                return arith_d.ConstantOp(
                    idx, ir.IntegerAttr.get(idx, node_or_val)
                ).result
            f32 = ir.F32Type.get()
            return arith_d.ConstantOp(
                f32, ir.FloatAttr.get(f32, float(node_or_val))
            ).result
        return None

    def _get_index_const(self, val: int) -> ir.Value:
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        idx = ir.IndexType.get()
        return arith_d.ConstantOp(idx, ir.IntegerAttr.get(idx, val)).result

    def _cast_to_index(self, val: ir.Value) -> ir.Value:
        """Cast an integer value to index type if needed."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        if isinstance(val.type, ir.IndexType):
            return val
        if isinstance(val.type, ir.RankedTensorType) and val.type.rank == 0:
            elem = val.type.element_type
            if isinstance(elem, (ir.IntegerType, ir.IndexType)):
                val = tensor_d.ExtractOp(val, []).result
                if isinstance(val.type, ir.IndexType):
                    return val
        return arith_d.IndexCastOp(ir.IndexType.get(), val).result

    def _shape_from_nodes(
        self, shape_nodes: list, operation_name: str = "op"
    ) -> list[int]:
        """Return a concrete integer shape list from a list of FX shape nodes.

        For ``_get_symnode("block_size_N")`` nodes the concrete block size is
        read from ``_block_id_to_size`` (which is populated before codegen
        starts).  For other nodes the ``meta["val"]`` SymInt hint is used as
        a fallback.
        """
        shape = []
        for s in shape_nodes:
            if isinstance(s, torch.fx.Node):
                tname = getattr(s.target, "__name__", "")
                if tname == "_get_symnode" and s.args:
                    key = s.args[0]
                    block_id = block_id_from_key(key)
                    if block_id is not None:
                        if block_id in self._block_id_to_size:
                            shape.append(self._block_id_to_size[block_id])
                            continue
                if tname in ("sym_size.int", "sym_size_int") and len(s.args) >= 2:
                    tensor_arg = s.args[0]
                    dim_arg = s.args[1]
                    if isinstance(tensor_arg, torch.fx.Node) and isinstance(
                        dim_arg, int
                    ):
                        tval = tensor_arg.meta.get("val")
                        if isinstance(tval, torch.Tensor):
                            from .aten_lowering import _resolve_dims

                            resolved = _resolve_dims(
                                tval.shape,
                                self._block_id_to_size,
                                self._block_hint_to_id,
                                self._block_symint_to_id,
                                self._block_id_to_upper_bound,
                            )
                            if 0 <= dim_arg < len(resolved):
                                shape.append(resolved[dim_arg])
                                continue
                # Fall back: use meta["val"] if it's concrete
                val = s.meta.get("val")
                if isinstance(val, torch.SymInt):
                    import sympy as _sympy

                    sym = str(val)
                    if sym.startswith("u") and sym[1:].isdigit():
                        block_id = int(sym[1:])
                        if block_id in self._block_id_to_size:
                            shape.append(self._block_id_to_size[block_id])
                            continue
                    expr = getattr(getattr(val, "node", None), "expr", None)
                    if isinstance(expr, _sympy.Symbol):
                        sym2 = str(expr)
                        if sym2.startswith("u") and sym2[1:].isdigit():
                            block_id = int(sym2[1:])
                            if block_id in self._block_id_to_size:
                                shape.append(self._block_id_to_size[block_id])
                                continue
                if val is not None:
                    try:
                        shape.append(int(val))
                        continue
                    except Exception:
                        pass
            elif isinstance(s, int):
                shape.append(s)
                continue
            shape.append(1)  # safe default
        return shape

    # ------------------------------------------------------------------
    # Individual node lowering methods
    # ------------------------------------------------------------------

    def _lower_host_tensor(self, node: torch.fx.Node) -> ir.Value | None:
        """``_host_tensor('name')`` → look up the function argument."""
        name = node.args[0]
        assert isinstance(name, str)
        if name in self._param_to_value:
            return self._param_to_value[name]

        # Alias fallback: some host-tensor names correspond to view/contiguous
        # aliases traced as separate host symbols (e.g. "z") while the backing
        # tensor ultimately originates from an argument (e.g. "x").
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            resolved = self._resolve_host_tensor_alias_value(val)
            if resolved is not None:
                aliased = self._materialize_host_tensor_alias_shape(resolved, node)
                if aliased is not None:
                    return aliased
                return resolved

        return None

    def _resolve_host_tensor_alias_value(self, t: torch.Tensor) -> ir.Value | None:
        """Resolve a host tensor alias to an existing argument MLIR value.

        Walks the tensor's base chain and checks ``tensor_to_origin`` names
        against ``_param_to_value``.
        """
        seen: set[int] = set()
        cur: torch.Tensor | None = t

        while isinstance(cur, torch.Tensor) and id(cur) not in seen:
            seen.add(id(cur))
            origin = self.hf.tensor_to_origin.get(cur)
            if origin is not None:
                host_name = origin.host_str()
                if host_name in self._param_to_value:
                    return self._param_to_value[host_name]
            cur = getattr(cur, "_base", None)

        return None

    def _materialize_host_tensor_alias_shape(
        self,
        base_val: ir.Value,
        alias_node: torch.fx.Node,
    ) -> ir.Value | None:
        """Materialize a static-shape host tensor alias when shape differs.

        Currently supports flatten-style aliases by emitting
        ``tensor.collapse_shape``.
        """
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        base_shape = None
        try:
            base_ty = ir.RankedTensorType(base_val.type)
            base_shape = [int(d) for d in base_ty.shape]
            elem_ty = base_ty.element_type
        except Exception:
            return None

        alias_shape = self._shape_from_node_meta(alias_node)
        if alias_shape is None or base_shape == alias_shape:
            return base_val

        if len(alias_shape) == 1:
            base_numel = 1
            for d in base_shape:
                base_numel *= d
            if base_numel != int(alias_shape[0]):
                return None
            result_ty = ir.RankedTensorType.get(alias_shape, elem_ty)
            reassociation = [list(range(len(base_shape)))]
            return tensor_d.CollapseShapeOp(result_ty, base_val, reassociation).result

        return None

    def _lower_get_symnode(self, node: torch.fx.Node) -> ir.Value:
        """``_get_symnode('block_size_N')`` → integer index constant."""
        from mlir.dialects import arith as arith_d
        import mlir.ir as ir

        key: str = node.args[0]
        # Parse "block_size_N" to get block_id N.
        block_id = block_id_from_key(key)
        if block_id is None:
            raise ValueNotFoundError(node, context=f"invalid block key: {key!r}")
        size = self._block_id_to_size.get(block_id, 0)
        idx = ir.IndexType.get()
        return arith_d.ConstantOp(idx, ir.IntegerAttr.get(idx, size)).result

    def _lower_tile_index(self, node: torch.fx.Node) -> ir.Value | None:
        """``tile.index`` → 1D tensor of global offsets for the tile."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        if not node.args:
            return None

        tile_arg = node.args[0]
        sym_to_block_id = self._build_sym_to_block_id()
        block_id = self._infer_block_id_from_index(tile_arg, sym_to_block_id)
        if block_id is None and self._for_block_id_stack:
            # In nested loops, symbolic metadata can be lost for tile.index.
            # Fall back to the innermost active scf.for block id.
            block_id = self._for_block_id_stack[-1]

        shape = self._shape_from_node_meta(node)
        if shape is None:
            shape = []
        if not shape:
            if block_id is not None and block_id in self._block_id_to_size:
                shape = [self._block_id_to_size[block_id]]
            else:
                return None

        # Prefer configured tile size over stale metadata for tile.index.
        # In nested loops, symbolic metadata can point to the wrong block and
        # inflate extents (e.g. 64 instead of 32), which then poisons helper
        # signatures and gather shapes.
        if block_id is not None and block_id in self._block_id_to_size and shape:
            shape[0] = int(self._block_id_to_size[block_id])

        if (
            block_id is not None
            and block_id in self._block_id_to_upper_bound
            and self._block_id_to_upper_bound[block_id] > 0
        ):
            shape[0] = min(shape[0], int(self._block_id_to_upper_bound[block_id]))

        if len(shape) != 1:
            raise UnsupportedOperationError(
                "tile_index",
                reason="Only 1D tile.index lowering is implemented",
            )

        elem_ty: ir.Type = ir.IndexType.get()
        meta_val = node.meta.get("val")
        if isinstance(meta_val, torch.Tensor):
            try:
                meta_ty = torch_dtype_to_mlir(meta_val.dtype)
                if isinstance(meta_ty, (ir.IntegerType, ir.IndexType)):
                    elem_ty = meta_ty
            except Exception:
                pass

        index_ty = ir.IndexType.get()
        result_ty = ir.RankedTensorType.get(shape, elem_ty)
        op = tensor_d.GenerateOp(result_ty, [])
        body = op.operation.regions[0].blocks.append(index_ty)

        if block_id is not None and block_id in self._block_id_to_iv:
            base = self._block_id_to_iv[block_id]
        else:
            base = self._get_index_const(0)

        with ir.InsertionPoint(body):
            iv = body.arguments[0]
            if isinstance(elem_ty, ir.IndexType):
                value = arith_d.AddIOp(base, iv).result
                tensor_d.YieldOp(value)
            else:
                base_int = arith_d.IndexCastOp(elem_ty, base).result
                iv_int = arith_d.IndexCastOp(elem_ty, iv).result
                value = arith_d.AddIOp(base_int, iv_int).result
                tensor_d.YieldOp(value)

        return op.result

    def _lower_mask_to(self, node: torch.fx.Node) -> ir.Value | None:
        """Conservatively forward masked tensors when the backend has no mask IR."""
        if not node.args:
            return None
        return self._get_value(node.args[0])

    def _lower_subscript(self, node: torch.fx.Node) -> ir.Value | None:
        from .subscript_ops import lower_subscript

        return lower_subscript(self, node)

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
        """``full(shape, fill_val, dtype)`` → ``tensor.empty`` + ``linalg.fill``.

        Raises
        ------
        ShapeError
            If shape is invalid
        DynamicShapeError
            If dynamic shapes are encountered
        """
        from mlir.dialects import arith as arith_d
        from mlir.dialects import linalg as linalg_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        # args: shape (list/tuple of symnodes), fill_value, dtype, device
        try:
            shape_nodes = node.args[0]
            fill_val = node.args[1]
            dtype = node.args[2] if len(node.args) > 2 else torch.float32

            # Extract concrete shape from shape nodes.
            # _get_symnode nodes carry the block_id in their first arg
            # ("block_size_N"); use _block_id_to_size for correct sizes.
            shape = self._shape_from_nodes(shape_nodes, "full")

            mlir_dtype = torch_dtype_to_mlir(dtype)
            empty = tensor_d.EmptyOp(shape, mlir_dtype).result
            fill_attr = (
                ir.FloatAttr.get(mlir_dtype, float(fill_val))
                if isinstance(mlir_dtype, ir.FloatType)
                else ir.IntegerAttr.get(mlir_dtype, int(fill_val))
            )
            fill_const = arith_d.ConstantOp(mlir_dtype, fill_attr).result
            return linalg_d.fill(fill_const, outs=[empty])
        except (ShapeError, DynamicShapeError):
            raise
        except Exception as e:
            raise NodeLoweringError(
                node,
                reason=str(e),
                recovery_hint="Check tensor shapes and dtypes",
            ) from e

    def _lower_zeros(self, node: torch.fx.Node) -> ir.Value:
        """``zeros(shape, dtype)`` → ``tensor.empty`` + ``linalg.fill(0)``.

        Raises
        ------
        ShapeError
            If shape is invalid
        """
        from mlir.dialects import arith as arith_d
        from mlir.dialects import linalg as linalg_d
        from mlir.dialects import tensor as tensor_d

        try:
            shape_nodes = node.args[0]
            dtype = node.args[1] if len(node.args) > 1 else torch.float32

            # Extract concrete shape from shape nodes.
            shape = self._shape_from_nodes(shape_nodes, "zeros")

            mlir_dtype = torch_dtype_to_mlir(dtype)
            empty = tensor_d.EmptyOp(shape, mlir_dtype).result
            zero_attr = get_zero_attr(dtype)
            zero = arith_d.ConstantOp(mlir_dtype, zero_attr).result
            return linalg_d.fill(zero, outs=[empty])
        except (ShapeError, DynamicShapeError):
            raise
        except Exception as e:
            raise NodeLoweringError(
                node,
                reason=str(e),
                recovery_hint="Check tensor shapes and dtypes",
            ) from e

    def _lower_for_loop(self, node: torch.fx.Node) -> ir.Value:
        """``_for_loop(body_id, block_ids, upper_bounds, iter_args)`` → ``scf.for``."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import linalg as linalg_d
        from mlir.dialects import scf as scf_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        body_graph_id: int = node.args[0]
        block_ids: list[int] = list(node.args[1])
        upper_bounds: list[object] = list(node.args[2])
        iter_arg_nodes = list(node.args[3])

        # We expect a single reduction dimension for now.
        assert len(block_ids) == 1 and len(upper_bounds) == 1, (
            f"Only single-dim reduction loops supported; got block_ids={block_ids}"
        )
        block_id = block_ids[0]
        ub_src = upper_bounds[0]

        ub_static: int | None = None
        ub_val: ir.Value | None = None

        if isinstance(ub_src, int):
            ub_static = int(ub_src)
        elif isinstance(ub_src, torch.fx.Node):
            ub_val = self._get_value(ub_src)
            if ub_val is None:
                meta_val = ub_src.meta.get("val")
                if isinstance(meta_val, torch.Tensor) and meta_val.numel() == 1:
                    try:
                        ub_static = int(meta_val.item())
                    except Exception:
                        ub_static = None
                elif isinstance(meta_val, (int, float)):
                    ub_static = int(meta_val)
        elif isinstance(ub_src, ir.Value):
            ub_val = ub_src

        if ub_static is None and ub_val is None:
            raise NodeLoweringError(
                node,
                reason=f"Unsupported loop upper bound type: {type(ub_src).__name__}",
                recovery_hint="Ensure loop bounds are integer constants or scalar tensor values",
            )

        if ub_val is not None:
            ub_val = self._cast_to_index(ub_val)

        ub_for_match = ub_static if ub_static is not None else None
        # Nested loops can sometimes reuse an outer block_id in FX metadata.
        # If that happens, pick a non-active block whose configured size best
        # matches this loop upper bound.
        if block_id in self._block_id_to_iv:
            candidates = [
                bid
                for bid, size in self._block_id_to_size.items()
                if bid not in self._block_id_to_iv
                and size > 0
                and ub_for_match is not None
                and ub_for_match % size == 0
            ]
            if candidates:
                block_id = max(candidates, key=lambda bid: self._block_id_to_size[bid])

        step = self._block_id_to_size.get(
            block_id,
            ub_static if ub_static is not None else 1,
        )

        device_ir = self.hf.device_ir
        body_graph_info = device_ir.graphs[body_graph_id]
        body_graph = body_graph_info.graph

        output_node = next(n for n in body_graph.nodes if n.op == "output")
        out_args = output_node.args[0]
        if not isinstance(out_args, (list, tuple)):
            out_args = [out_args]

        # Build iter args from loop arguments. Some traced loops pass invariant
        # values (e.g. jagged row start/end tensors) in iter_arg_nodes even
        # though only a trailing subset is truly loop-carried.
        iter_pairs = [(a, self._get_value(a)) for a in iter_arg_nodes]
        iter_pairs = [(a, v) for a, v in iter_pairs if v is not None]

        carried_count = len(iter_pairs)
        if 0 < len(out_args) <= len(iter_pairs):
            carried_count = len(out_args)

        invariant_pairs = iter_pairs[: len(iter_pairs) - carried_count]
        carried_pairs = iter_pairs[len(iter_pairs) - carried_count :]
        iter_init_vals = [v for _, v in carried_pairs]

        active_outer_block_ids = set(self._block_id_to_iv.keys())

        synthetic_store_ctx: dict[str, Any] | None = None
        synthetic_iter_index: int | None = None

        # If this loop body directly stores tiles, synthesize one tile tensor
        # iter_arg so values produced in the loop body dominate their later use
        # in forall.in_parallel. This is needed even when other iter args are
        # already carried through the loop.
        store_nodes = [
            n
            for n in body_graph.nodes
            if n.op == "call_function" and getattr(n.target, "__name__", "") == "store"
        ]
        if len(store_nodes) == 1:
            store_node = store_nodes[0]
            target_node = store_node.args[0]
            index_nodes = store_node.args[1]
            value_node = store_node.args[2]

            target_val = self._get_value(target_node)
            value_meta = (
                value_node.meta.get("val")
                if isinstance(value_node, torch.fx.Node)
                else None
            )

            target_type: ir.RankedTensorType | None = None
            target_rank_matches = True
            if target_val is not None:
                target_type = ir.RankedTensorType(target_val.type)
                target_rank_matches = target_type.rank == len(index_nodes)

            if isinstance(index_nodes, (list, tuple)) and target_rank_matches:
                if target_val is not None:
                    assert target_type is not None
                    full_shape = list(target_type.shape)
                    elem_ty = target_type.element_type
                else:
                    if isinstance(value_meta, torch.Tensor):
                        value_shape = [int(d) for d in value_meta.shape]
                        elem_ty = torch_dtype_to_mlir(value_meta.dtype)
                    else:
                        value_shape = [1 for _ in index_nodes]
                        elem_ty = torch_dtype_to_mlir(torch.float32)
                    full_shape = list(value_shape)
                rank = len(full_shape)
                sym_to_block_id = self._build_sym_to_block_id()
                dim_block_ids: list[int | None] = []
                inner_dim: int | None = None

                for dim, idx_node in enumerate(index_nodes):
                    if dim >= rank:
                        break
                    dim_bid = self._infer_block_id_from_index(idx_node, sym_to_block_id)
                    dim_block_ids.append(dim_bid)
                    # If metadata aliases both dims to the same block_id,
                    # prefer the last occurrence (inner-most index dim).
                    if dim_bid == block_id:
                        inner_dim = dim

                # Fallback when metadata doesn't preserve exact block-id
                # mapping for every dimension in nested loops.
                while len(dim_block_ids) < rank:
                    dim_block_ids.append(None)
                if inner_dim is None:
                    inner_dim = min(rank - 1, max(0, len(index_nodes) - 1))

                if inner_dim is not None:
                    tile_shape: list[int] = []
                    flush_offsets: list[ir.Value] = []
                    outer_bids = [
                        bid for bid in active_outer_block_ids if bid != block_id
                    ]
                    fallback_outer_bid = outer_bids[0] if outer_bids else None
                    for dim, dim_size in enumerate(full_shape):
                        dim_bid = dim_block_ids[dim]
                        if dim == inner_dim or dim_bid == block_id:
                            tile_shape.append(
                                ub_static if ub_static is not None else step
                            )
                            flush_offsets.append(self._get_index_const(0))
                        elif (
                            isinstance(dim_bid, int)
                            and dim_bid in active_outer_block_ids
                            and dim_bid in self._block_id_to_size
                        ):
                            tile_shape.append(self._block_id_to_size[dim_bid])
                            flush_offsets.append(self._block_id_to_iv[dim_bid])
                        elif (
                            fallback_outer_bid is not None
                            and fallback_outer_bid in self._block_id_to_size
                        ):
                            tile_shape.append(
                                self._block_id_to_size[fallback_outer_bid]
                            )
                            flush_offsets.append(
                                self._block_id_to_iv[fallback_outer_bid]
                            )
                        else:
                            tile_shape.append(int(dim_size))
                            flush_offsets.append(self._get_index_const(0))

                    tile_empty = tensor_d.EmptyOp(
                        tile_shape,
                        elem_ty,
                    ).result
                    if isinstance(elem_ty, ir.FloatType):
                        zero_attr = ir.FloatAttr.get(elem_ty, 0.0)
                    else:
                        zero_attr = ir.IntegerAttr.get(elem_ty, 0)
                    zero = arith_d.ConstantOp(elem_ty, zero_attr).result
                    tile_init = linalg_d.fill(zero, outs=[tile_empty])

                    synthetic_iter_index = len(iter_init_vals)
                    iter_init_vals.append(tile_init)
                    synthetic_store_ctx = {
                        "block_id": block_id,
                        "inner_dim": inner_dim,
                        "rank": rank,
                        "flush_offsets": flush_offsets,
                        "current": None,
                    }

        lb_val = self._get_index_const(0)
        ub_val = (
            ub_val
            if ub_val is not None
            else self._get_index_const(ub_static if ub_static is not None else step)
        )
        step_val = self._get_index_const(step)

        for_op = scf_d.ForOp(lb_val, ub_val, step_val, iter_args=iter_init_vals)
        body_block = for_op.body

        with ir.InsertionPoint(body_block):
            # Register the for loop IV for this block dim.
            # Save the previous value (the outer forall IV) so we can restore it
            # after the loop — the outer IV is still needed for the store offsets
            # that come after the scf.for in the forall body.
            previous_iv = self._block_id_to_iv.get(block_id)
            for_iv = body_block.arguments[0]
            self._block_id_to_iv[block_id] = for_iv
            self._for_block_id_stack.append(block_id)

            # Map iter_arg placeholder nodes to the for body's iter args.
            placeholders = [n for n in body_graph.nodes if n.op == "placeholder"]
            if len(placeholders) > len(iter_pairs):
                placeholders = placeholders[-len(iter_pairs) :]

            invariant_placeholders = placeholders[: len(invariant_pairs)]
            for ph_node, (_, inv_val) in zip(
                invariant_placeholders, invariant_pairs, strict=False
            ):
                self._node_to_value[ph_node] = inv_val

            carried_placeholders = placeholders[len(invariant_pairs) :]
            for ph_node, body_arg in zip(
                carried_placeholders, body_block.arguments[1:], strict=False
            ):
                self._node_to_value[ph_node] = body_arg

            if synthetic_store_ctx is not None and synthetic_iter_index is not None:
                synthetic_store_ctx["current"] = body_block.arguments[
                    1 + synthetic_iter_index
                ]
                self._for_store_ctx_stack.append(synthetic_store_ctx)

            # Process the body graph.
            self._process_graph(body_graph)

            # Collect the output values for scf.yield.
            yield_vals = []
            for a in out_args:
                v = self._get_value(a) if isinstance(a, torch.fx.Node) else None
                if v is not None:
                    yield_vals.append(v)

            if synthetic_store_ctx is not None:
                current = synthetic_store_ctx.get("current")
                if current is not None:
                    insert_at = (
                        synthetic_iter_index
                        if synthetic_iter_index is not None
                        else len(yield_vals)
                    )
                    if insert_at <= len(yield_vals):
                        yield_vals.insert(insert_at, current)
                    else:
                        yield_vals.append(current)

            if len(yield_vals) != len(iter_init_vals):
                if len(yield_vals) > len(iter_init_vals):
                    raise NodeLoweringError(
                        node,
                        reason=(
                            "Loop body yielded more values than iter_args: "
                            f"{len(yield_vals)} > {len(iter_init_vals)}"
                        ),
                        recovery_hint="Ensure loop-carried values match loop iter_args",
                    )
                passthrough_count = len(iter_init_vals) - len(yield_vals)
                passthrough_vals = list(body_block.arguments[1 : 1 + passthrough_count])
                yield_vals = yield_vals + passthrough_vals

            scf_d.YieldOp(yield_vals)

            if (
                synthetic_store_ctx is not None
                and self._for_store_ctx_stack
                and self._for_store_ctx_stack[-1] is synthetic_store_ctx
            ):
                self._for_store_ctx_stack.pop()

            if self._for_block_id_stack and self._for_block_id_stack[-1] == block_id:
                self._for_block_id_stack.pop()

        # Restore the outer IV for block_id so subsequent store offsets (which
        # use the outer forall IVs) resolve to the correct values.
        if previous_iv is not None:
            self._block_id_to_iv[block_id] = previous_iv
        elif block_id in self._block_id_to_iv:
            del self._block_id_to_iv[block_id]

        if synthetic_store_ctx is not None and synthetic_iter_index is not None:
            final_tile = for_op.results[synthetic_iter_index]
            self._forall_insert_slices.append(
                (final_tile, synthetic_store_ctx["flush_offsets"])
            )

        # Return the for result (a list containing the final iter_arg value).
        # We return the ForOp itself; callers unpack via getitem.
        return for_op  # type: ignore[return-value]

    def _lower_getitem(self, node: torch.fx.Node) -> ir.Value | None:
        from .memory_ops import lower_getitem

        return lower_getitem(self, node)

    def _lower_load(self, node: torch.fx.Node) -> ir.Value:
        """``load(tensor, index_list)`` → ``tensor.extract_slice``."""
        from mlir.dialects import arith as arith_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        tensor_node = node.args[0]
        index_nodes = node.args[1]  # list of symnode values (tile sizes)

        tensor_val = self._get_value(tensor_node)
        assert tensor_val is not None, f"No value for tensor node {tensor_node}"

        # The index_nodes are the SIZES of the tile.  Each size corresponds to
        # one tensor dimension.  The OFFSET comes from the loop IV for that
        # dimension.
        tensor_type = ir.RankedTensorType(tensor_val.type)
        ndim = len(tensor_type.shape)

        # Advanced indexing fast-path: gather from a 1D source tensor using
        # an N-D integer index tensor. This is required for jagged flatten-
        # then-gather patterns where hl.load(x_flat, [flat_indices]) should
        # produce a tensor with the same shape as flat_indices.
        if ndim == 1 and len(index_nodes) == 1:
            gather_index_val = self._get_value(index_nodes[0])
            if gather_index_val is not None:
                try:
                    gather_index_ty = ir.RankedTensorType(gather_index_val.type)
                except Exception:
                    gather_index_ty = None
                if gather_index_ty is not None and gather_index_ty.rank >= 1:
                    from .load_ops import lower_flat_gather

                    gathered = lower_flat_gather(
                        self,
                        tensor_node,
                        tensor_val,
                        gather_index_val,
                        gather_index_ty,
                        tensor_type,
                    )
                    if gathered is not None:
                        return gathered

        # The load index list tells us which block dims map to which tensor dim.
        # We use the sympy symbol of each index to find the block_id.
        offsets: list[ir.Value] = []
        sizes: list[int] = []
        strides: list[int] = []

        # Build a mapping from sympy symbol string (e.g. "u0") → block_id.
        sym_to_block_id = self._build_sym_to_block_id()
        used_block_ids: set[int] = set()

        # In synthetic inner-loop store mode, nested tile metadata can lose the
        # outer/inner block-id distinction (both dimensions may look like the
        # inner block). Use the active context to force a stable mapping.
        forced_dim_block_id: dict[int, int] = {}
        if self._for_store_ctx_stack:
            ctx = self._for_store_ctx_stack[-1]
            inner_block_id = int(ctx.get("block_id", -1))
            inner_dim = int(ctx.get("inner_dim", -1))
            rank = int(ctx.get("rank", ndim))
            if 0 <= inner_dim < rank:
                forced_dim_block_id[inner_dim] = inner_block_id
                outer_candidates = [
                    bid for bid in self._block_id_to_iv if bid != inner_block_id
                ]
                if len(outer_candidates) == 1:
                    outer_bid = outer_candidates[0]
                    for d in range(rank):
                        if d != inner_dim:
                            forced_dim_block_id[d] = outer_bid

        # Derive tile sizes from the load result's meta — identical source to
        # what preprocess_aten_nodes uses, guaranteeing consistent shapes.
        from .aten_lowering import _resolve_dims

        result_val = node.meta.get("val")
        result_sizes: list[int] | None = None
        if isinstance(result_val, torch.Tensor):
            result_sizes = _resolve_dims(
                result_val.shape,
                self._block_id_to_size,
                self._block_hint_to_id,
                self._block_symint_to_id,
            )

        for dim, idx_node in enumerate(index_nodes):
            if dim >= ndim:
                break
            index_extent: int | None = None
            if isinstance(idx_node, torch.fx.Node):
                idx_val = self._get_value(idx_node)
                if idx_val is not None:
                    try:
                        idx_ty = ir.RankedTensorType(idx_val.type)
                        if idx_ty.rank == 1:
                            index_extent = int(idx_ty.shape[0])
                    except Exception:
                        pass
            forced = dim in forced_dim_block_id
            block_id, index_bias = self._infer_index_block_and_bias(
                idx_node, sym_to_block_id
            )
            allow_fallback_inference = isinstance(idx_node, torch.fx.Node)
            if forced:
                block_id = forced_dim_block_id[dim]
                index_bias = 0
            if (
                allow_fallback_inference
                and block_id is None
                and result_sizes is not None
                and dim < len(result_sizes)
            ):
                # Fallback: infer IV by matching this tile dimension size to
                # exactly one active block size in the current loop nest.
                dim_size = result_sizes[dim]
                matching = [
                    bid
                    for bid in self._block_id_to_iv
                    if self._block_id_to_size.get(bid) == dim_size
                    and bid not in used_block_ids
                ]
                if len(matching) == 1:
                    block_id = matching[0]
            if allow_fallback_inference and block_id is None:
                dim_extent = int(tensor_type.shape[dim])
                candidates = [
                    bid
                    for bid in self._block_id_to_iv
                    if bid not in used_block_ids
                    and self._block_id_to_upper_bound.get(bid) == dim_extent
                ]
                if len(candidates) == 1:
                    block_id = candidates[0]
            if block_id is not None and block_id in self._block_id_to_iv:
                offset_val = self._block_id_to_iv[block_id]
                if index_bias != 0:
                    offset_val = arith_d.AddIOp(
                        offset_val,
                        self._get_index_const(index_bias),
                    ).result
                offsets.append(offset_val)
                used_block_ids.add(block_id)
            else:
                # No IV registered → offset is 0.
                offsets.append(self._get_index_const(0))
            # Size: prefer configured block sizes when the dimension maps to a
            # known tile block_id; metadata can be ambiguous in nested loops.
            dim_extent = int(tensor_type.shape[dim])
            if block_id is not None and block_id in self._block_id_to_size:
                configured = self._block_id_to_size.get(block_id, dim_extent)
                upper = self._block_id_to_upper_bound.get(block_id)
                if upper is not None and upper > 0:
                    configured = min(configured, int(upper))

                if (
                    (not forced)
                    and result_sizes is not None
                    and dim < len(result_sizes)
                ):
                    sizes.append(min(configured, result_sizes[dim], dim_extent))
                else:
                    sizes.append(min(configured, dim_extent))
            else:
                if index_extent is not None:
                    sizes.append(min(index_extent, dim_extent))
                elif result_sizes is not None and dim < len(result_sizes):
                    sizes.append(min(result_sizes[dim], dim_extent))
                else:
                    sizes.append(dim_extent)
            strides.append(1)

        # Result type.
        elem_ty = tensor_type.element_type
        result_type = ir.RankedTensorType.get(sizes, elem_ty)

        return tensor_d.ExtractSliceOp(
            result_type,
            tensor_val,
            offsets,
            [],  # no dynamic sizes
            [],  # no dynamic strides
            static_offsets=[ir.ShapedType.get_dynamic_size()] * len(offsets),
            static_sizes=sizes,
            static_strides=strides,
        ).result

    def _lower_store(self, node: torch.fx.Node) -> None:
        from .memory_ops import lower_store

        lower_store(self, node)

    def _lower_store_node(self, node: torch.fx.Node) -> ir.Value | None:
        self._lower_store(node)
        return None

    # ------------------------------------------------------------------
    # ATen pre-pass: lower all ATen nodes before codegen starts
    # ------------------------------------------------------------------

    def _prebuild_aten_helpers(self, module: ir.Module) -> None:
        """Scan device IR for ATen nodes, lower them all via one torch-mlir pass.

        Results are stored in ``self._node_to_aten_func`` and the helper
        ``func.func`` operations are inserted at the module's top level.
        """
        from .aten_lowering import is_aten_op
        from .aten_lowering import preprocess_aten_nodes

        # Propagate symbolic shapes from outer _for_loop iter-args into inner
        # loop body placeholders BEFORE scanning ATen nodes.  Without this,
        # placeholder shapes in nested loop bodies are evaluated to their hint
        # values (e.g. both tile_m and tile_n evaluate to 64), making them
        # indistinguishable when resolving block_ids in _resolve_shape.
        self._restore_symbolic_shapes_in_bodies()
        self._refresh_aten_tensor_meta()

        aten_nodes: list[torch.fx.Node] = []
        for graph_info in self.hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if is_aten_op(node):
                    target_name = str(node.target)
                    target_overload = getattr(node.target, "__name__", "")
                    if (
                        "aten.addmm" in target_name
                        or "aten.mm" in target_name
                        or "aten.matmul" in target_name
                        or "aten.bmm" in target_name
                        or "aten.baddbmm" in target_name
                        or target_overload
                        in (
                            "addmm.default",
                            "mm.default",
                            "matmul.default",
                            "bmm.default",
                            "baddbmm.default",
                        )
                    ):
                        # These are lowered directly in codegen.
                        continue
                    aten_nodes.append(node)

        if not aten_nodes:
            return

        self._node_to_aten_func = preprocess_aten_nodes(
            aten_nodes,
            module,
            self._block_id_to_size,
            self._block_hint_to_id,
            self._block_symint_to_id,
            self._block_id_to_upper_bound,
        )

    def _helper_signature_matches(
        self,
        func_name: str,
        input_mlir_vals: list[ir.Value],
    ) -> bool:
        """Return True when helper function arg types match provided MLIR values."""
        import mlir.ir as ir

        if self._mlir_module is None:
            return False

        for op in self._mlir_module.body.operations:
            name_attr = op.attributes.get("sym_name")
            if name_attr is None:
                continue
            name = name_attr.value if hasattr(name_attr, "value") else str(name_attr)
            if name != func_name:
                continue
            ftype = ir.FunctionType(ir.TypeAttr(op.attributes["function_type"]).value)
            if len(ftype.inputs) != len(input_mlir_vals):
                return False
            return all(
                str(expected) == str(actual.type)
                for expected, actual in zip(ftype.inputs, input_mlir_vals, strict=True)
            )

        return False

    def _helper_is_identity(self, func_name: str) -> bool:
        """Return True if helper body is `return arg0` with no intermediate ops."""
        if self._mlir_module is None:
            return False

        for op in self._mlir_module.body.operations:
            name_attr = op.attributes.get("sym_name")
            if name_attr is None:
                continue
            name = name_attr.value if hasattr(name_attr, "value") else str(name_attr)
            if name != func_name:
                continue

            try:
                block = op.regions[0].blocks[0]
                ops = list(block.operations)
                if len(ops) != 1:
                    return False
                ret = ops[0]
                if ret.operation.name != "func.return":
                    return False
                if len(ret.operands) != 1 or len(block.arguments) != 1:
                    return False
                return ret.operands[0] == block.arguments[0]
            except Exception:
                return False

        return False

    def _rebuild_aten_helper_for_call(
        self,
        node: torch.fx.Node,
        input_mlir_vals: list[ir.Value],
    ) -> tuple[str, list[ir.Type]] | None:
        """Build a call-site-specific ATen helper variant using current operand types."""
        from .aten_lowering import collect_tensor_input_positions
        from .aten_lowering import normalized_aten_args
        from .aten_lowering import preprocess_aten_nodes

        if self._mlir_module is None:
            return None

        import mlir.ir as ir

        norm_args = normalized_aten_args(node)
        tensor_positions = collect_tensor_input_positions(node)
        if len(tensor_positions) != len(input_mlir_vals):
            return None

        from .type_utils import mlir_dtype_to_torch

        backups: list[tuple[torch.fx.Node, object, object]] = []
        override_by_position: dict[int, torch.Tensor] = {}
        try:
            for arg_idx, mlir_val in zip(
                tensor_positions, input_mlir_vals, strict=True
            ):
                arg_node = norm_args[arg_idx]
                if not isinstance(arg_node, torch.fx.Node):
                    continue

                rty = ir.RankedTensorType(mlir_val.type)
                shape = [int(d) for d in rty.shape]
                elem_key = str(rty.element_type)
                dtype = mlir_dtype_to_torch(
                    elem_key,
                    default=torch.int64 if elem_key == "index" else torch.float32,
                )
                if elem_key not in {
                    "f32",
                    "f64",
                    "f16",
                    "bf16",
                    "i1",
                    "i8",
                    "i16",
                    "i32",
                    "i64",
                    "index",
                }:
                    return None

                old_val = arg_node.meta.get("val")
                old_tm = arg_node.meta.get("tensor_meta")
                backups.append((arg_node, old_val, old_tm))

                concrete = torch.zeros(shape, dtype=dtype)
                arg_node.meta["val"] = concrete
                override_by_position[arg_idx] = concrete

            try:
                rebuilt_map = preprocess_aten_nodes(
                    [node],
                    self._mlir_module,
                    self._block_id_to_size,
                    self._block_hint_to_id,
                    self._block_symint_to_id,
                    self._block_id_to_upper_bound,
                    {id(node): override_by_position},
                )
            except Exception as exc:
                log.warning(
                    "On-demand helper rebuild failed for node %s (%s): %s",
                    node.name,
                    node.target,
                    exc,
                )
                return None
            rebuilt = rebuilt_map.get(id(node))
            if rebuilt is not None:
                self._node_to_aten_func[id(node)] = rebuilt
            return rebuilt
        finally:
            for arg_node, old_val, old_tm in backups:
                if old_val is None:
                    arg_node.meta.pop("val", None)
                else:
                    arg_node.meta["val"] = old_val
                if old_tm is None:
                    arg_node.meta.pop("tensor_meta", None)
                else:
                    arg_node.meta["tensor_meta"] = old_tm

    def _refresh_aten_tensor_meta(self) -> None:
        from .aten_prepass import refresh_aten_tensor_meta

        refresh_aten_tensor_meta(self.hf)

    def _restore_symbolic_shapes_in_bodies(self) -> None:
        """Copy symbolic meta['val'] from outer iter-args into inner body placeholders.

        Inside a ``_for_loop`` body graph, placeholder meta may be evaluated to
        hint integers instead of keeping the original symbolic SymInts (e.g.
        str='u0').  Restoring them ensures _resolve_shape can distinguish blocks
        by symbolic name rather than falling back to ambiguous hint values.

        Additionally registers the iter-arg tensor's SymInt shape dimensions in
        ``_block_symint_to_id`` so identity-based lookup works even when the
        acc tensor's SymInts are different objects from the _get_symnode ones.
        """
        for graph_info in self.hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                if getattr(node.target, "__name__", "") != "_for_loop":
                    continue
                body_graph_id: int = node.args[0]
                iter_arg_outer_nodes = list(node.args[3])  # outer-scope FX nodes
                body_graph = self.hf.device_ir.graphs[body_graph_id].graph
                body_phs = [n for n in body_graph.nodes if n.op == "placeholder"]
                for ph, outer_node in zip(body_phs, iter_arg_outer_nodes, strict=False):
                    if not isinstance(outer_node, torch.fx.Node):
                        continue
                    outer_val = outer_node.meta.get("val")
                    if not isinstance(outer_val, torch.Tensor):
                        continue
                    upper_bounds = node.args[2] if len(node.args) > 2 else None
                    # Register the outer tensor's SymInt shape dimensions so
                    # _resolve_shape can match them by identity.  Match each
                    # shape dimension to its block_id via the outer node's
                    # shape-arg nodes (e.g. _get_symnode("block_size_0")).
                    shape_arg = outer_node.args[0] if outer_node.args else None
                    if isinstance(shape_arg, (list, tuple)):
                        for i, shape_node in enumerate(shape_arg):
                            if (
                                isinstance(shape_node, torch.fx.Node)
                                and getattr(shape_node.target, "__name__", "")
                                == "_get_symnode"
                                and shape_node.args
                                and isinstance(shape_node.args[0], str)
                                and "block_size_" in shape_node.args[0]
                                and i < len(outer_val.shape)
                            ):
                                block_id = block_id_from_key(shape_node.args[0])
                                if block_id is None:
                                    continue
                                dim_symint = outer_val.shape[i]
                                if isinstance(dim_symint, torch.SymInt):
                                    import sympy as _sympy

                                    expr = getattr(
                                        getattr(dim_symint, "node", None), "expr", None
                                    )
                                    if isinstance(expr, _sympy.Symbol):
                                        self._block_symint_to_id[id(expr)] = block_id
                    # Build a concrete-shaped fake tensor using config block sizes.
                    # This replaces the ambiguous symbolic meta so _resolve_shape
                    # sees plain Python integers and maps them correctly.
                    if isinstance(shape_arg, (list, tuple)):
                        concrete_shape = self._shape_from_nodes(
                            list(shape_arg), "iter_arg"
                        )
                        # Tile dimensions in loop bodies must be bounded by the
                        # loop upper bounds; block sizes can be larger.
                        if isinstance(upper_bounds, (list, tuple)):
                            for i, ub in enumerate(upper_bounds):
                                if i >= len(concrete_shape):
                                    break
                                try:
                                    ub_int = int(ub)
                                except Exception:
                                    continue
                                if ub_int > 0:
                                    concrete_shape[i] = min(concrete_shape[i], ub_int)
                        concrete_val = torch.zeros(
                            concrete_shape, dtype=outer_val.dtype
                        )
                        ph.meta["val"] = concrete_val
                        # Propagate to _new_var aliases and update all nodes that
                        # derive their shape from the placeholder (sym_size.int, loads).
                        for body_node in body_graph.nodes:
                            if body_node.op != "call_function":
                                continue
                            tname = getattr(body_node.target, "__name__", "")
                            if (
                                tname == "_new_var"
                                and body_node.args
                                and body_node.args[0] is ph
                            ):
                                body_node.meta["val"] = concrete_val
                            elif tname in ("sym_size.int", "sym_size_int"):
                                tensor_arg = (
                                    body_node.args[0] if body_node.args else None
                                )
                                dim_arg = (
                                    body_node.args[1]
                                    if len(body_node.args) > 1
                                    else None
                                )
                                if (
                                    tensor_arg is ph
                                    and isinstance(dim_arg, int)
                                    and dim_arg < len(concrete_shape)
                                ):
                                    body_node.meta["val"] = concrete_shape[dim_arg]
                            elif tname == "load":
                                # Recompute load result shape using _shape_from_nodes
                                # so extract_slice sizes match the config values.
                                load_index_nodes = (
                                    body_node.args[1]
                                    if len(body_node.args) > 1
                                    else None
                                )
                                if load_index_nodes is not None:
                                    try:
                                        load_shape = self._shape_from_nodes(
                                            list(load_index_nodes), "load"
                                        )
                                        old_val = body_node.meta.get("val")
                                        if isinstance(old_val, torch.Tensor) and len(
                                            load_shape
                                        ) == len(old_val.shape):
                                            body_node.meta["val"] = torch.zeros(
                                                load_shape, dtype=old_val.dtype
                                            )
                                    except Exception:
                                        pass
                    else:
                        ph.meta["val"] = outer_val
                        for body_node in body_graph.nodes:
                            if (
                                body_node.op == "call_function"
                                and getattr(body_node.target, "__name__", "")
                                == "_new_var"
                                and body_node.args
                                and body_node.args[0] is ph
                            ):
                                body_node.meta["val"] = outer_val

    # ------------------------------------------------------------------
    # Helper: infer block_id from an index FX node
    # ------------------------------------------------------------------

    def _build_sym_to_block_id(self) -> dict[str, int]:
        """Build mapping from sympy symbol name (e.g. "u0") → block_id."""
        mapping: dict[str, int] = {}
        for bs in self.env.block_sizes:
            # The symbolic variable name follows the pattern "uN" where N == block_id.
            mapping[f"u{bs.block_id}"] = bs.block_id
        return mapping

    def _infer_block_id_from_index(
        self,
        idx_node: object,
        sym_to_block_id: dict[str, int],
    ) -> int | None:
        block_id, _ = self._infer_index_block_and_bias(idx_node, sym_to_block_id)
        return block_id

    def _infer_index_block_and_bias(
        self,
        idx_node: object,
        sym_to_block_id: dict[str, int],
    ) -> tuple[int | None, int]:
        """Infer ``(block_id, additive_bias)`` for simple tile-index expressions."""

        block_id = self._infer_block_id_from_index_symbolic(idx_node, sym_to_block_id)
        if block_id is not None:
            return block_id, 0

        if isinstance(idx_node, int):
            return None, int(idx_node)

        if not isinstance(idx_node, torch.fx.Node):
            return None, 0

        target_name = str(idx_node.target)
        tname = getattr(idx_node.target, "__name__", "")
        is_add = "aten.add" in target_name or tname in ("add.Tensor", "add.default")
        if is_add and len(idx_node.args) >= 2:
            left_block, left_bias = self._infer_index_block_and_bias(
                idx_node.args[0], sym_to_block_id
            )
            right_block, right_bias = self._infer_index_block_and_bias(
                idx_node.args[1], sym_to_block_id
            )

            if left_block is not None and right_block is None:
                return left_block, left_bias + right_bias
            if right_block is not None and left_block is None:
                return right_block, right_bias + left_bias

            # Shape-based fallback when symbolic provenance is lost through
            # arithmetic wrappers (common for tile.index + constant patterns).
            if left_block is None and right_block is None:
                left_shape_block = self._infer_block_id_from_value_shape(
                    idx_node.args[0]
                )
                right_shape_block = self._infer_block_id_from_value_shape(
                    idx_node.args[1]
                )
                if left_shape_block is not None and right_shape_block is None:
                    return left_shape_block, left_bias + right_bias
                if right_shape_block is not None and left_shape_block is None:
                    return right_shape_block, right_bias + left_bias

        return None, 0

    def _infer_block_id_from_value_shape(self, idx_node: object) -> int | None:
        """Infer block_id from a 1D tensor index extent when symbolic info is unavailable."""
        if not isinstance(idx_node, torch.fx.Node):
            return None
        val = self._get_value(idx_node)
        if val is None:
            return None

        import mlir.ir as ir

        try:
            val_ty = ir.RankedTensorType(val.type)
        except Exception:
            return None

        if val_ty.rank != 1:
            return None

        extent = int(val_ty.shape[0])
        candidates = [
            bid
            for bid in self._block_id_to_iv
            if (
                self._block_id_to_upper_bound.get(bid) == extent
                or self._block_id_to_size.get(bid) == extent
            )
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _infer_block_id_from_index_symbolic(
        self,
        idx_node: object,
        sym_to_block_id: dict[str, int],
    ) -> int | None:
        """Given an FX node representing a tile index, return its block_id.

        We look at the ``meta["val"]`` of the node, which is a ``torch.SymInt``
        backed by a sympy symbol of the form ``uN``.
        """
        if not isinstance(idx_node, torch.fx.Node):
            return None
        val = idx_node.meta.get("val")
        if val is None:
            return None
        if isinstance(val, torch.SymInt):
            sym_str = str(val)
            if sym_str in sym_to_block_id:
                return sym_to_block_id[sym_str]
            # Fallback for cases where symbol strings are unavailable/rewritten.
            import sympy as _sympy

            val_expr = getattr(getattr(val, "node", None), "expr", None)
            if (
                isinstance(val_expr, _sympy.Symbol)
                and id(val_expr) in self._block_symint_to_id
            ):
                return self._block_symint_to_id[id(val_expr)]
            if hasattr(idx_node, "target"):
                tname = getattr(idx_node.target, "__name__", "")
                if tname == "_get_symnode" and idx_node.args:
                    key = idx_node.args[0]
                    if isinstance(key, str) and "block_size_" in key:
                        return block_id_from_key(key)
        # For sym_size.int nodes the val is a SymInt too.
        if isinstance(val, torch.SymInt):
            tname = getattr(idx_node.target, "__name__", "")
            if tname in ("sym_size.int", "sym_size_int"):
                tensor_node = idx_node.args[0]
                dim_idx = int(idx_node.args[1])
                tensor_val = (
                    tensor_node.meta.get("val")
                    if isinstance(tensor_node, torch.fx.Node)
                    else None
                )
                if isinstance(tensor_val, torch.Tensor):
                    shape_val = tensor_val.shape[dim_idx]
                    if isinstance(shape_val, torch.SymInt):
                        sv_expr = getattr(
                            getattr(shape_val, "node", None), "expr", None
                        )
                        if (
                            isinstance(sv_expr, _sympy.Symbol)
                            and id(sv_expr) in self._block_symint_to_id
                        ):
                            return self._block_symint_to_id[id(sv_expr)]
                        sym_str = str(shape_val)
                        if sym_str in sym_to_block_id:
                            return sym_to_block_id[sym_str]
        return None
