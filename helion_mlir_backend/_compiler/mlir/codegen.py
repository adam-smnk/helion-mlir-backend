"""Core MLIR module builder for Helion kernels.

Walks the :class:`~helion._compiler.device_ir.DeviceIR` produced by the
standard Helion compilation pipeline and emits an ``mlir.ir.Module`` using
Linalg-on-Tensors abstraction.

Mapping summary
---------------
Helion construct          → MLIR construct
──────────────────────────────────────────────────────────────────────────────
Outer ``hl.tile([m,n])``  → normalized ``scf.forall`` + ``shared_outs``
Inner ``hl.tile(k)``      → ``scf.for``   (sequential, carries accumulator)
``hl.zeros([bm,bn])``     → ``tensor.empty`` + ``linalg.fill``
``hl.load(t, idx)``       → ``tensor.extract_slice``
``hl.store(t, idx, v)``   → ``tensor.parallel_insert_slice`` (in forall term.)
matmul family, ``hl.dot``,
einsum, ``acc + mm``      → one ``linalg.matmul``/``batch_matmul``/``contract``
Pointwise aten ops        → ``linalg.generic`` or ``arith.*``
``_host_tensor(name)``    → reference to the corresponding function argument
``_get_symnode(bs_N)``    → the concrete block-size integer constant
``_phi``                  → the result of the enclosing ``scf.for``
``_new_var``              → pass-through (same MLIR value)

ATen lowering architecture
--------------------------
Every node is dispatched by target identity through ``lowering/registry.py``.
Helion-specific device-IR nodes determine the tile/control-flow structure and
lower directly to ``scf``/``tensor``/``linalg`` operations. Generic ATen nodes
are extracted into pure FX subgraphs by ``aten_lowering.py``, imported through
``torch_mlir.extras.fx_importer.FxImporter``, lowered through torch-mlir's
Torch-to-Linalg pipeline, and cloned back into this module as private helper
``func.func`` operations. A small set of direct ATen lowerings remains for
Helion-specific operand conventions and contractions.

This split is necessary because ``FxImporter`` cannot import Helion's
non-ATen FX targets, while torch-mlir provides broad, reusable coverage for
the ordinary ATen portions of each tile body.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import functools
import logging
from typing import TYPE_CHECKING

from mlir.dialects import func as func_d
import mlir.ir as ir
import torch
import torch.fx

from .aten_bridge import AtenHelperTable
from .build_context import BuildContext
from .lowering import build_phase_body
from .lowering import lower_node
from .support import MLIRBackendError
from .support import UnsupportedOperationError
from .support import torch_tensor_to_mlir_type

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction

log = logging.getLogger(__name__)

_shared_mlir_context: ir.Context | None = None
_context_construction_count = 0


def _get_shared_mlir_context() -> ir.Context:
    """Return a process-wide ``mlir.ir.Context``, created lazily on first use.

    Constructing a fresh ``ir.Context()`` per compile is unsafe: the native
    bindings spawn a background thread pool and load every dialect on each
    construction, and repeated create/destroy cycles across kernel compiles
    (e.g. one process compiling many configs) have been observed to segfault
    when a still-shutting-down context's threads race with a newly
    constructed one. MLIR contexts are designed to host many independent
    modules, so reusing a single one for the process avoids the race.
    """
    global _shared_mlir_context, _context_construction_count
    if _shared_mlir_context is None:
        _context_construction_count += 1
        if _context_construction_count > 1:
            # A second construction means _shared_mlir_context was reset to
            # None by something other than this function -- exactly the
            # crash pattern this cache exists to prevent (see docstring).
            raise RuntimeError(
                "The process-wide shared mlir.ir.Context was reconstructed. "
                "This previously caused segfaults from racing background "
                "thread pools across kernel compiles. Route all MLIR context "
                "access through _get_shared_mlir_context() and never "
                "construct ir.Context() directly."
            )
        _shared_mlir_context = ir.Context()
    return _shared_mlir_context


@dataclass
class PhaseModuleResult:
    """One compiled phase, ready for the direct-call driver to JIT and run.

    ``input_names``/``output_names`` are host variable names, in the exact
    order the phase's ``func.func`` expects its arguments/produces its
    results -- the driver looks up real tensors by these names and binds
    results back under ``output_names`` for later phases to consume.
    """

    name: str
    module: ir.Module
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


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
        self.context.lower_node_callback = functools.partial(lower_node, self.context)

    def build(self) -> ir.Module:
        """Build and return the generated MLIR module."""

        try:
            ctx = _get_shared_mlir_context()

            with ir.Location.unknown(ctx):
                module = ir.Module.create()
                self.context.mlir_module = module
                self.context.mlir_context = ctx
                self.context.aten_helpers = AtenHelperTable(module)
                with ir.InsertionPoint(module.body), self.hf:
                    self._resolve_geometry()
                    self._prebuild_aten_helpers(module)
                    self._build_function()
            return module
        except MLIRBackendError:
            raise
        except Exception as exc:
            exc.add_note(f"While generating MLIR for Helion kernel '{self.hf.name}'")
            raise

    def build_phase_modules(self) -> list[PhaseModuleResult]:
        """Build one MLIR module per ``hl.barrier()``-separated phase.

        Used only by the direct ``backend="mlir"`` driver path
        (``bound_kernel.py::mlir_compile_config``) -- never by the public
        ``generate_mlir()``/``execute_mlir()`` two-call flow, which keeps
        using :meth:`build`'s single-module, single-phase-only contract.
        """

        from .phase_plan import build_phase_plans
        from .phase_plan import find_extra_host_tensor_names
        from .phase_plan import find_host_tensor_fake_value
        from .phase_plan import iter_phase_graphs
        from .phase_plan import resolve_host_variable_name

        tensor_params = [
            (name, value)
            for name, value in self.hf.params.arguments.items()
            if isinstance(value, torch.Tensor)
        ]
        declared_param_names = {name for name, _ in tensor_params}
        extra_names = find_extra_host_tensor_names(self.hf, declared_param_names)
        plans = build_phase_plans(self.hf, declared_param_names, extra_names)

        name_to_fake: dict[str, torch.Tensor] = dict(tensor_params)
        for name in extra_names:
            fake = find_host_tensor_fake_value(self.hf, name)
            if fake is not None:
                name_to_fake[name] = fake

        results: list[PhaseModuleResult] = []
        try:
            mlir_ctx = _get_shared_mlir_context()

            with ir.Location.unknown(mlir_ctx), self.hf:
                self._resolve_geometry()

                for plan in plans:
                    module = ir.Module.create()
                    self.context.mlir_module = module
                    self.context.mlir_context = mlir_ctx
                    self.context.aten_helpers = AtenHelperTable(module)
                    phase_name = f"{self.hf.name}__phase{plan.phase_index}"
                    phase_graphs = iter_phase_graphs(self.hf, plan.root_ids)

                    with ir.InsertionPoint(module.body):
                        self._prebuild_aten_helpers(module, graphs=phase_graphs)

                        input_types = [
                            torch_tensor_to_mlir_type(name_to_fake[name])
                            for name in plan.input_names
                        ]
                        output_types = [
                            torch_tensor_to_mlir_type(tensor)
                            for _, tensor in plan.outputs
                        ]
                        fn = func_d.FuncOp(
                            phase_name, ir.FunctionType.get(input_types, output_types)
                        )
                        fn.attributes["sym_visibility"] = ir.StringAttr.get("public")
                        entry = fn.add_entry_block()
                        with ir.InsertionPoint(entry):
                            self.context.reset_for_new_function()
                            for name, arg in zip(
                                plan.input_names, entry.arguments, strict=True
                            ):
                                self.context.param_to_value[name] = arg

                            result_vals = build_phase_body(
                                self.context,
                                plan.root_ids,
                                [name for name, _ in plan.outputs],
                            )
                            func_d.ReturnOp(result_vals)

                    output_names: list[str] = []
                    for _, tensor in plan.outputs:
                        resolved = resolve_host_variable_name(self.hf, tensor)
                        if resolved is None:
                            raise UnsupportedOperationError(
                                "unresolvable phase output name",
                                reason=(
                                    "a phase's output tensor must be bound to a "
                                    "plain host-level variable name to be threaded "
                                    "to a later phase or the kernel's return "
                                    "statement"
                                ),
                            )
                        name_to_fake[resolved] = tensor
                        output_names.append(resolved)

                    results.append(
                        PhaseModuleResult(
                            name=phase_name,
                            module=module,
                            input_names=list(plan.input_names),
                            output_names=output_names,
                        )
                    )
            return results
        except MLIRBackendError:
            raise
        except Exception as exc:
            exc.add_note(f"While generating MLIR for Helion kernel '{self.hf.name}'")
            raise

    def _build_function(self) -> None:
        from .phase_plan import requires_multi_phase_driver
        from .support import UnsupportedOperationError

        tensor_params = [
            (name, value)
            for name, value in self.hf.params.arguments.items()
            if isinstance(value, torch.Tensor)
        ]
        needs_driver, extra_names = requires_multi_phase_driver(self.hf, tensor_params)
        multi_phase = len(self.hf.device_ir.phases) > 1

        if not needs_driver:
            self._build_single_phase_function(tensor_params)
            return

        # Multi-phase (hl.barrier()) and/or host-tensor-interop kernels are
        # only supported through the direct backend="mlir" call path (see
        # bound_kernel.py::mlir_compile_config -> build_phase_modules()),
        # which drives a real host-side wrapper between phases. This
        # single-module entry point (generate_mlir()/execute_mlir()) has no
        # such driver, so it can't supply extra host tensors or thread real
        # values between phases.
        if multi_phase:
            reason = "this kernel uses hl.barrier() (multiple phases)"
        else:
            reason = (
                f"this kernel depends on host tensor(s) {extra_names!r} not "
                "in its declared parameters"
            )
        raise UnsupportedOperationError(
            "multi-phase or host-tensor-interop kernel",
            reason=(
                f"{reason}; generate_mlir()/execute_mlir() only support "
                "single-phase kernels whose device loops reference only "
                "declared parameters"
            ),
            alternatives=[
                "call the kernel directly via @helion.kernel(backend='mlir')"
            ],
        )

    def _build_single_phase_function(
        self, tensor_params: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Inputs: tensor params not written, or also read. Outputs: every written tensor."""
        effects = self.context.effects
        graphs = [graph_info.graph for graph_info in self.hf.device_ir.graphs]
        outputs = effects.written_in(graphs)
        read = effects.read_in(graphs)
        output_types = [torch_tensor_to_mlir_type(effects.fakes[n]) for n in outputs]
        input_params = [
            (name, value)
            for name, value in tensor_params
            if name not in outputs or name in read
        ]
        input_types = [torch_tensor_to_mlir_type(value) for _, value in input_params]

        fn = func_d.FuncOp(self.hf.name, ir.FunctionType.get(input_types, output_types))
        fn.attributes["sym_visibility"] = ir.StringAttr.get("public")
        entry = fn.add_entry_block()
        with ir.InsertionPoint(entry):
            for (name, _), arg in zip(input_params, entry.arguments, strict=True):
                self.context.param_to_value[name] = arg
            roots = range(len(self.hf.device_ir.root_ids))
            func_d.ReturnOp(build_phase_body(self.context, list(roots), outputs))

    def _resolve_geometry(self) -> None:
        """Record loop geometry, tensor effects and contractions (needs ``with self.hf:``)."""
        from .analysis.contractions import ContractionPlan
        from .analysis.geometry import KernelGeometry
        from .analysis.tensor_effects import TensorEffects

        self.context.effects = TensorEffects.from_host_function(self.hf)
        self.context.geometry = KernelGeometry.from_host_function(
            self.hf, self.config, self.env
        )
        self.context.contractions = ContractionPlan.from_graphs(
            [graph_info.graph for graph_info in self.hf.device_ir.graphs]
        )

    def _prebuild_aten_helpers(
        self, module: ir.Module, graphs: list[torch.fx.Graph] | None = None
    ) -> None:
        """Lower every ATen node without a direct lowering via one torch-mlir pass.

        Helper ``func.func`` operations are inserted at the module's top level and
        recorded in ``self.context.aten_helpers``. Scoped to *graphs* when given
        (one phase's own graphs), otherwise the whole kernel's.
        """
        from .aten_lowering import is_aten_op
        from .aten_lowering import preprocess_aten_nodes
        from .einsum_capture import is_einsum_node
        from .support import refresh_aten_tensor_meta
        from .support import restore_symbolic_shapes_in_bodies

        # Nested loop-body placeholders must carry their outer symbolic shapes
        # before ATen nodes are scanned (see symbolic_shape_restoration).
        restore_symbolic_shapes_in_bodies(self.hf, self.context)
        refresh_aten_tensor_meta(self.hf)

        search_graphs = (
            graphs
            if graphs is not None
            else [gi.graph for gi in self.hf.device_ir.graphs]
        )
        plan = self.context.contractions
        aten_nodes = [
            node
            for graph in search_graphs
            for node in graph.nodes
            if is_aten_op(node)
            and not is_einsum_node(node)
            and plan.needs_helper(node)
            and not self.context.has_symint_operand(node)
        ]
        if not aten_nodes:
            return

        entries = preprocess_aten_nodes(
            aten_nodes,
            module,
            self.context.geometry.block_sizes(),
            self.env,
            self.context.geometry.spans(),
        )
        self.context.aten_helpers.replace(entries)
