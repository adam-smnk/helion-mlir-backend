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

from .dynamic_shapes import SymbolTable
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

        # Populated during build():
        # Maps FX node → MLIR value (ir.Value)
        self._node_to_value: dict[torch.fx.Node, ir.Value] = {}
        # Maps parameter name → function argument MLIR value
        self._param_to_value: dict[str, ir.Value] = {}
        # Maps block_id → concrete integer size
        self._block_id_to_size: dict[int, int] = {}
        # hint_value → block_id; last resort for concretized SymInts
        self._block_hint_to_id: dict[int, int] = {}
        # id(sympy.Symbol) → block_id; stable identity via cached sympy symbols
        self._block_symint_to_id: dict[int, int] = {}
        # Maps block_id → current MLIR offset Value (loop IV)
        self._block_id_to_iv: dict[int, ir.Value] = {}
        # Stack of pending insert_slice ops collected inside forall.in_parallel
        self._forall_insert_slices: list[tuple] = []  # (value, offsets)

        # Symbol table for tracking dynamic shapes (SymInt values)
        self._symbol_table = SymbolTable()

        # These are populated inside build() once the ir.Module/Context exist.
        self._mlir_module: ir.Module | None = None
        self._mlir_context: ir.Context | None = None
        # Pre-computed ATen helper map: id(node) → (func_name, return_types).
        # Populated by _prebuild_aten_helpers() before _build_function() runs.
        self._node_to_aten_func: dict[int, tuple[str, list]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> ir.Module:
        """Build and return the MLIR module.

        Returns
        -------
        ir.Module
            The generated MLIR module

        Raises
        ------
        ModuleBuilderError
            If module building fails at any stage
        """
        import mlir.ir as ir

        try:
            ctx = ir.Context()
            # Register dialects we use
            from mlir.dialects import arith as arith_d  # noqa: F401
            from mlir.dialects import (
                func as func_d,  # noqa: F401 – side-effect registration
            )
            from mlir.dialects import linalg as linalg_d  # noqa: F401
            from mlir.dialects import scf as scf_d  # noqa: F401
            from mlir.dialects import tensor as tensor_d  # noqa: F401

            with ir.Location.unknown(ctx):
                module = ir.Module.create()
                self._mlir_module = module
                self._mlir_context = ctx
                with ir.InsertionPoint(module.body), self.hf:
                    # Resolve block sizes first (needed by both phases).
                    self._resolve_block_sizes()
                    # Phase 1: lower all ATen nodes once, insert helpers at
                    # module top level (before the main kernel function).
                    self._prebuild_aten_helpers(module)
                    # Phase 2: build the main kernel function.
                    self._build_function()
            return module
        except (
            ModuleBuilderError,
            NodeLoweringError,
            ValueNotFoundError,
            UnsupportedOperationError,
        ):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise ModuleBuilderError(
                "module_creation",
                reason=str(e),
                recovery_hint="Check that kernel has static_shapes=True and all ops are in hl.tile() loops",
            ) from e

    # ------------------------------------------------------------------
    # Function signature
    # ------------------------------------------------------------------

    def _build_function(self) -> None:
        from mlir.dialects import func as func_d
        import mlir.ir as ir

        hf = self.hf
        # Collect tensor parameters (non-constexpr, non-symbolic)
        tensor_params: list[tuple[str, torch.Tensor]] = []
        for name, val in hf.params.arguments.items():
            if isinstance(val, torch.Tensor):
                tensor_params.append((name, val))

        # Build input types
        input_types: list[ir.Type] = [
            torch_tensor_to_mlir_type(t) for _, t in tensor_params
        ]

        # Output type: the return tensor.
        # Helion kernels return a tensor assigned to the last name in
        # ``tensor_params`` that appears in the kernel return statement.
        # For static_shapes=True, the output tensor is among the fake_args
        # or is created inside the kernel (torch.empty).  We detect it as the
        # tensor registered with name "out" (or the last one if none found).
        #
        # TODO(helion-mlir): handle multiple return tensors.
        out_name, out_tensor = self._find_output_tensor(tensor_params)
        output_types: list[ir.Type] = [torch_tensor_to_mlir_type(out_tensor)]

        # Filter tensor_params to INPUTS only (exclude the output tensor which
        # is created inside the kernel body, not passed as an argument).
        input_params = [(n, t) for n, t in tensor_params if n != out_name]
        input_types = [torch_tensor_to_mlir_type(t) for _, t in input_params]

        func_type = ir.FunctionType.get(input_types, output_types)
        fn = func_d.FuncOp(hf.name, func_type)
        fn.attributes["sym_visibility"] = ir.StringAttr.get("public")

        entry = fn.add_entry_block()
        with ir.InsertionPoint(entry):
            # Register parameter names → MLIR values
            for (name, _), arg in zip(input_params, entry.arguments, strict=True):
                self._param_to_value[name] = arg

            # Block sizes are pre-resolved in build() before this call.
            # Only resolve here if they haven't been populated yet (e.g. in
            # tests that call _build_function() directly, without HostFunction).
            if not self._block_id_to_size:
                with self.hf:
                    self._resolve_block_sizes()

            # Build the kernel body
            result = self._build_kernel_body(out_tensor)
            func_d.ReturnOp([result])

    def _find_output_tensor(
        self, tensor_params: list[tuple[str, torch.Tensor]]
    ) -> tuple[str, torch.Tensor]:
        """Find the name and fake value of the output tensor.

        The output tensor is the one created by ``torch.empty`` inside the
        kernel.  In the HostFunction params it appears with the name used in
        the kernel body (commonly ``"out"``).  We look for it via
        ``_host_tensor`` nodes in the device IR that reference tensors NOT
        passed as inputs.
        """

        # Collect tensor_param names that are genuine INPUTS (referenced in
        # the original kernel signature ``hf.args``).
        input_names = {arg.arg for arg in self.hf.args.args}

        # Find any tensor param NOT in the input list - that's our output.
        for name, tensor in tensor_params:
            if name not in input_names:
                return name, tensor

        # Fallback: look through device IR for _host_tensor calls on tensors
        # not in input_names.
        for g in self.hf.device_ir.graphs:
            for node in g.graph.nodes:
                if (
                    node.op == "call_function"
                    and getattr(node.target, "__name__", "") == "_host_tensor"
                ):
                    tname = node.args[0]
                    if tname not in input_names:
                        val = node.meta.get("val")
                        if isinstance(val, torch.Tensor):
                            return tname, val

        # Last resort: use the last tensor param.
        return tensor_params[-1]

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
                    if isinstance(key, str) and "block_size_" in key:
                        block_id = int(key.split("_")[-1])
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

    # ------------------------------------------------------------------
    # Kernel body – outer forall structure
    # ------------------------------------------------------------------

    def _build_kernel_body(self, out_tensor: torch.Tensor) -> ir.Value:
        """Build the kernel body and return the final result value."""
        from mlir.dialects import scf as scf_d
        from mlir.dialects import tensor as tensor_d
        import mlir.ir as ir

        device_ir = self.hf.device_ir
        grid_block_ids: list[int] = []
        for ids in device_ir.grid_block_ids:
            grid_block_ids.extend(ids)

        # Build concrete bounds and steps for the forall loop.
        out_shape = [int(d) for d in out_tensor.shape]
        # Upper bounds from tensor shape; lower bounds = 0.
        lbs = [0] * len(grid_block_ids)
        ubs = [out_shape[i] for i in range(len(grid_block_ids))]
        steps = [self._block_id_to_size[bid] for bid in grid_block_ids]

        # Create the output tensor (shared_outs for forall).
        out_empty = tensor_d.EmptyOp(
            out_shape, torch_dtype_to_mlir(out_tensor.dtype)
        ).result

        forall = scf_d.ForallOp(lbs, ubs, steps, shared_outs=[out_empty])

        # Register grid block IVs → forall induction variables.
        ivs = list(forall.induction_variables)
        for bid, iv in zip(grid_block_ids, ivs, strict=True):
            self._block_id_to_iv[bid] = iv

        # Build the forall body.
        body_block = forall.body
        with ir.InsertionPoint(body_block):
            # The last block arg is the iter arg (shared output tile).
            shared_out_arg = next(iter(forall.inner_iter_args))

            # Process all root graphs.
            self._process_root_graphs(shared_out_arg)

            # in_parallel terminator with parallel_insert_slice ops.
            # Hoist all arith.constant ops BEFORE the in_parallel region —
            # scf.forall.in_parallel may only contain ParallelCombiningOpInterface
            # ops (i.e. tensor.parallel_insert_slice), not arith.constant.
            in_parallel = scf_d.InParallelOp()
            with ir.InsertionPoint(in_parallel.block):
                for value, offsets in self._forall_insert_slices:
                    # Sizes and strides are fully static: read from the source
                    # tensor type rather than dynamic SSA values.
                    src_type = ir.RankedTensorType(value.type)
                    static_sizes = list(src_type.shape)
                    ndim = len(static_sizes)
                    tensor_d.ParallelInsertSliceOp(
                        value,
                        shared_out_arg,
                        offsets,  # dynamic offsets (forall IVs)
                        [],  # no dynamic sizes — all static from src
                        [],  # no dynamic strides — all static 1
                        static_offsets=[ir.ShapedType.get_dynamic_size()] * ndim,
                        static_sizes=static_sizes,
                        static_strides=[1] * ndim,
                    )

        return forall.results[0]

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

            # --- Helion tracing ops ---
            if tname == "_host_tensor":
                return self._lower_host_tensor(node)
            if tname == "_get_symnode":
                return self._lower_get_symnode(node)
            if tname == "_new_var":
                # Alias / SSA rename – just pass through.
                return self._get_value(node.args[0])
            if tname == "_phi":
                # After the for loop: _phi(init, loop_result) → loop_result.
                return self._get_value(node.args[1])
            if tname == "_for_loop":
                return self._lower_for_loop(node)
            if tname == "getitem":
                # Extract element from a list/tuple result.
                return self._lower_getitem(node)

            # --- Memory ops ---
            if tname == "load":
                return self._lower_load(node)
            if tname == "store":
                self._lower_store(node)
                return None

            # --- Creation ops ---
            if tname == "full":
                return self._lower_full(node)
            if tname == "zeros":
                return self._lower_zeros(node)

            # --- Tensor shape ops ---
            if tname in ("sym_size.int", "sym_size_int"):
                return self._lower_sym_size(node)

            # Special-case addmm in nested reductions: lowering directly to
            # linalg.matmul with the accumulator as the `outs` tensor preserves
            # loop-carried equivalence required by downstream lighthouse passes.
            target_name = str(node.target)
            if "aten.addmm" in target_name:
                lowered_addmm = self._lower_aten_addmm(node)
                if lowered_addmm is not None:
                    return lowered_addmm
            if "aten.add.Tensor" in target_name:
                lowered_add_matmul = self._lower_aten_add_matmul_accumulate(node)
                if lowered_add_matmul is not None:
                    return lowered_add_matmul

            # --- All standard ATen ops → pre-built linalg helper (func.call) ---
            from mlir.dialects import func as func_d

            from .aten_lowering import collect_tensor_input_positions
            from .aten_lowering import is_aten_op
            from .aten_lowering import normalized_aten_args

            if is_aten_op(node):
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
        from mlir.dialects import linalg as linalg_d

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
            if "aten.mm" in second_name or "aten.matmul" in second_name:
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

        return linalg_d.matmul(lhs, rhs, outs=[acc])

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
        import mlir.ir as ir

        if isinstance(val.type, ir.IndexType):
            return val
        return arith_d.IndexCastOp(ir.IndexType.get(), val).result

    def _extract_concrete_shape(
        self,
        shape_nodes: list[Any],
        operation_name: str = "unknown",
    ) -> list[int]:
        """Extract concrete shape from potentially dynamic shape nodes.

        This method uses the symbol table to track SymInt values and
        returns a concrete shape suitable for MLIR code generation.

        Parameters
        ----------
        shape_nodes : list[Any]
            Nodes/values representing each dimension
        operation_name : str
            Name of operation (for logging)

        Returns
        -------
        list[int]
            Concrete shape values (SymInts resolved to their values or defaults)
        """
        from .dynamic_shapes import extract_symbol_from_shape

        shape: list[int] = []
        for i, s in enumerate(shape_nodes):
            sym_name, concrete = extract_symbol_from_shape(s, i, self._symbol_table)
            if sym_name is not None and concrete is None:
                # Symbol couldn't be resolved, use default
                log.debug(
                    "Dynamic dim %d in %s: symbol=%s (using default)",
                    i,
                    operation_name,
                    sym_name,
                )
                concrete = 1  # Safe default
            shape.append(max(1, concrete or 0))
        return shape

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
                    if isinstance(key, str) and "block_size_" in key:
                        block_id = int(key.split("_")[-1])
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
                            from .aten_lowering import (
                                _resolve_shape as _aten_resolve_shape,
                            )

                            resolved = _aten_resolve_shape(
                                tval,
                                self._block_id_to_size,
                                self._block_hint_to_id,
                                self._block_symint_to_id,
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
        block_id = int(key.split("_")[-1])
        size = self._block_id_to_size.get(block_id, 0)
        idx = ir.IndexType.get()
        return arith_d.ConstantOp(idx, ir.IntegerAttr.get(idx, size)).result

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
        from mlir.dialects import scf as scf_d
        import mlir.ir as ir

        body_graph_id: int = node.args[0]
        block_ids: list[int] = list(node.args[1])
        upper_bounds: list[int] = list(node.args[2])
        iter_arg_nodes = list(node.args[3])

        # We expect a single reduction dimension for now.
        assert len(block_ids) == 1 and len(upper_bounds) == 1, (
            f"Only single-dim reduction loops supported; got block_ids={block_ids}"
        )
        block_id = block_ids[0]
        ub = int(upper_bounds[0])
        # Nested loops can sometimes reuse an outer block_id in FX metadata.
        # If that happens, pick a non-active block whose configured size best
        # matches this loop upper bound.
        if block_id in self._block_id_to_iv:
            candidates = [
                bid
                for bid, size in self._block_id_to_size.items()
                if bid not in self._block_id_to_iv and size > 0 and ub % size == 0
            ]
            if candidates:
                block_id = max(candidates, key=lambda bid: self._block_id_to_size[bid])

        step = self._block_id_to_size.get(block_id, ub)

        # Build iter_args list.
        iter_init_vals = [self._get_value(a) for a in iter_arg_nodes]
        iter_init_vals = [v for v in iter_init_vals if v is not None]

        lb_val = self._get_index_const(0)
        ub_val = self._get_index_const(ub)
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

            # Map iter_arg placeholder nodes to the for body's iter args.
            device_ir = self.hf.device_ir
            body_graph_info = device_ir.graphs[body_graph_id]
            body_graph = body_graph_info.graph
            placeholders = [n for n in body_graph.nodes if n.op == "placeholder"]
            for ph_node, body_arg in zip(
                placeholders, body_block.arguments[1:], strict=True
            ):
                self._node_to_value[ph_node] = body_arg

            # Process the body graph.
            self._process_graph(body_graph)

            # Collect the output values for scf.yield.
            output_node = next(n for n in body_graph.nodes if n.op == "output")
            out_args = output_node.args[0]
            if not isinstance(out_args, (list, tuple)):
                out_args = [out_args]
            yield_vals = []
            for a in out_args:
                v = self._get_value(a) if isinstance(a, torch.fx.Node) else None
                if v is not None:
                    yield_vals.append(v)

            scf_d.YieldOp(yield_vals)

        # Restore the outer IV for block_id so subsequent store offsets (which
        # use the outer forall IVs) resolve to the correct values.
        if previous_iv is not None:
            self._block_id_to_iv[block_id] = previous_iv
        elif block_id in self._block_id_to_iv:
            del self._block_id_to_iv[block_id]

        # Return the for result (a list containing the final iter_arg value).
        # We return the ForOp itself; callers unpack via getitem.
        return for_op  # type: ignore[return-value]

    def _lower_getitem(self, node: torch.fx.Node) -> ir.Value | None:
        """``getitem(for_op, index)`` → extract one result from a for loop."""
        container = node.args[0]
        idx = int(node.args[1])
        container_val = self._get_value(container)
        if container_val is None:
            return None
        # For scf.ForOp, results are accessed via .results[idx].
        if hasattr(container_val, "results"):
            return container_val.results[idx]
        # For a plain list/tuple result already stored as ir.Value.
        return container_val

    def _lower_load(self, node: torch.fx.Node) -> ir.Value:
        """``load(tensor, index_list)`` → ``tensor.extract_slice``."""
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

        # The load index list tells us which block dims map to which tensor dim.
        # We use the sympy symbol of each index to find the block_id.
        offsets: list[ir.Value] = []
        sizes: list[int] = []
        strides: list[int] = []

        # Build a mapping from sympy symbol string (e.g. "u0") → block_id.
        sym_to_block_id = self._build_sym_to_block_id()
        used_block_ids: set[int] = set()

        # Derive tile sizes from the load result's meta — identical source to
        # what preprocess_aten_nodes uses, guaranteeing consistent shapes.
        from .aten_lowering import _resolve_shape as _aten_resolve_shape

        result_val = node.meta.get("val")
        result_sizes: list[int] | None = None
        if isinstance(result_val, torch.Tensor):
            result_sizes = _aten_resolve_shape(
                result_val,
                self._block_id_to_size,
                self._block_hint_to_id,
                self._block_symint_to_id,
            )

        for dim, idx_node in enumerate(index_nodes):
            if dim >= ndim:
                break
            block_id = self._infer_block_id_from_index(idx_node, sym_to_block_id)
            if (
                block_id is None
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
            if block_id is not None and block_id in self._block_id_to_iv:
                offsets.append(self._block_id_to_iv[block_id])
                used_block_ids.add(block_id)
            else:
                # No IV registered → offset is 0.
                offsets.append(self._get_index_const(0))
            # Size: prefer result shape (same as ATen helper signatures),
            # then block_id lookup, then full tensor dimension as last resort.
            if result_sizes is not None and dim < len(result_sizes):
                sizes.append(result_sizes[dim])
            elif block_id is not None:
                sizes.append(
                    self._block_id_to_size.get(block_id, int(tensor_type.shape[dim]))
                )
            else:
                sizes.append(int(tensor_type.shape[dim]))
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
        """``store(tensor, index_list, value)`` → record for forall.in_parallel."""
        import mlir.ir as ir

        index_nodes = node.args[1]
        value_node = node.args[2]

        value = self._get_value(value_node)
        assert value is not None, f"No value for store value node {value_node}"

        ndim = len(ir.RankedTensorType(value.type).shape)

        sym_to_block_id = self._build_sym_to_block_id()
        offsets: list[ir.Value] = []

        for dim, idx_node in enumerate(index_nodes):
            if dim >= ndim:
                break
            block_id = self._infer_block_id_from_index(idx_node, sym_to_block_id)
            if block_id is not None and block_id in self._block_id_to_iv:
                offsets.append(self._block_id_to_iv[block_id])
            else:
                offsets.append(self._get_index_const(0))

        # Queue the insert_slice for the forall.in_parallel region.
        self._forall_insert_slices.append((value, offsets))

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

        aten_nodes: list[torch.fx.Node] = []
        for graph_info in self.hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if is_aten_op(node):
                    aten_nodes.append(node)

        if not aten_nodes:
            return

        self._node_to_aten_func = preprocess_aten_nodes(
            aten_nodes,
            module,
            self._block_id_to_size,
            self._block_hint_to_id,
            self._block_symint_to_id,
        )

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
                                block_id = int(shape_node.args[0].split("_")[-1])
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
                        return int(key.split("_")[-1])
            try:
                hint_val = int(val)
                if hint_val in self._block_hint_to_id:
                    return self._block_hint_to_id[hint_val]
            except (TypeError, ValueError):
                pass
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
                        try:
                            hint_val = int(shape_val)
                            if hint_val in self._block_hint_to_id:
                                return self._block_hint_to_id[hint_val]
                        except (TypeError, ValueError):
                            pass
        return None
