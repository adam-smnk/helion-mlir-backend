"""Analyze a compiled kernel's phases and host-tensor dependencies.

Read-only analysis, no MLIR/codegen side effects: this module figures out,
for a kernel using ``hl.barrier()`` and/or host-computed tensors beyond its
own declared parameters, what each phase needs as input and produces as
output -- as plain host variable names -- so a future driver can compile one
MLIR function per phase and call them in order, threading real tensors
between phases by name (see ``host_prefix.py`` for how those real values are
obtained).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch
import torch.fx

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction


@dataclass
class PhasePlan:
    """One ``hl.barrier()``-separated phase's compilation inputs/outputs."""

    phase_index: int
    root_ids: list[int]
    input_names: list[str]
    outputs: list[tuple[str, torch.Tensor]] = field(default_factory=list)


def find_extra_host_tensor_names(
    hf: HostFunction, declared_param_names: set[str]
) -> list[str]:
    """Every distinct ``_host_tensor(name)`` reference that isn't a declared
    parameter or a view/reshape alias of one (same alias-resolution logic as
    ``lowering/host_tensor_ops.py::lower_host_tensor``, reused here without
    needing a full ``BuildContext``).

    ``hf.device_ir.graphs`` is a flat list of every graph -- root graphs and
    every nested ``_for_loop`` body -- so scanning it covers all nesting
    depths without a separate recursive walk.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for graph_info in hf.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "_host_tensor":
                continue
            name = node.args[0]
            if (
                not isinstance(name, str)
                or name in seen
                or name in declared_param_names
            ):
                continue
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and _aliases_declared_param(
                hf, value, declared_param_names
            ):
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def _aliases_declared_param(
    hf: HostFunction, tensor: torch.Tensor, declared_param_names: set[str]
) -> bool:
    """Walk a tensor's origin/``._base`` chain for a declared-parameter match."""
    seen_ids: set[int] = set()
    current: torch.Tensor | None = tensor
    while isinstance(current, torch.Tensor) and id(current) not in seen_ids:
        seen_ids.add(id(current))
        origin = hf.tensor_to_origin.get(current)
        if origin is not None and origin.host_str() in declared_param_names:
            return True
        current = getattr(current, "_base", None)
    return False


def resolve_host_variable_name(hf: HostFunction, tensor: torch.Tensor) -> str | None:
    """The plain host-level identifier a tensor is bound to, if resolvable.

    Only a simple ``ast.Name``-style origin (``host_str()`` is a valid
    identifier) can be threaded between phases by the driver; anything else
    (a computed expression origin) is not yet supported.
    """
    origin = hf.tensor_to_origin.get(tensor)
    if origin is None:
        return None
    name = origin.host_str()
    return name if name.isidentifier() else None


def find_host_tensor_fake_value(hf: HostFunction, name: str) -> torch.Tensor | None:
    """The traced ``FakeTensor`` for a ``_host_tensor(name)`` reference.

    Used to determine an extra host tensor's dtype/shape at compile time
    (the real value is only available later, from the host-prefix driver).
    """
    for graph_info in hf.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "__name__", "") == "_host_tensor"
                and node.args[0] == name
            ):
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    return value
    return None


def requires_multi_phase_driver(
    hf: HostFunction, tensor_params: list[tuple[str, torch.Tensor]]
) -> tuple[bool, list[str]]:
    """Whether *hf* needs the phase-module driver instead of a single,
    ``generate_mlir()``-style module.

    Returns ``(needed, extra_host_tensor_names)``. A kernel's own resolved
    output tensor(s) are excluded from ``extra_host_tensor_names`` -- they're
    always referenced via ``_host_tensor(name)`` too (to get the destination
    to store into), which would otherwise make every single-phase kernel
    look like it needs a phase driver.
    """
    from .output_resolver import OutputTensorResolver

    declared_param_names = {name for name, _ in tensor_params}
    multi_phase = len(hf.device_ir.phases) > 1
    out_params = OutputTensorResolver(hf).resolve_all(tensor_params)
    out_tensor_ids = {id(t) for _, t in out_params}
    candidate_names = find_extra_host_tensor_names(hf, declared_param_names)
    extra_names = [
        name
        for name in candidate_names
        if id(find_host_tensor_fake_value(hf, name)) not in out_tensor_ids
    ]
    return multi_phase or bool(extra_names), extra_names


def iter_phase_graphs(hf: HostFunction, root_ids: list[int]) -> list[torch.fx.Graph]:
    """Every graph reachable from *root_ids*, including nested loop bodies.

    ``root_ids`` (``KernelPhase.roots``) are POSITIONS into
    ``device_ir.root_ids``, not graph ids directly -- ``device_ir.root_ids``
    itself holds the real graph id for each position (nested-loop bodies can
    be registered before a later root, so a root's graph id and its position
    frequently differ).
    """
    device_ir = hf.device_ir
    graphs: list[torch.fx.Graph] = []
    stack = [device_ir.graphs[device_ir.root_ids[rid]].graph for rid in root_ids]
    while stack:
        graph = stack.pop()
        graphs.append(graph)
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "__name__", "") == "_for_loop"
            ):
                stack.append(device_ir.graphs[node.args[0]].graph)
    return graphs


def _phase_host_tensor_names(hf: HostFunction, root_ids: list[int]) -> list[str]:
    """Every distinct ``_host_tensor(name)`` referenced within one phase."""
    seen: set[str] = set()
    ordered: list[str] = []
    for graph in iter_phase_graphs(hf, root_ids):
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "_host_tensor":
                continue
            name = node.args[0]
            if isinstance(name, str) and name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def build_phase_plans(
    hf: HostFunction,
    declared_param_names: set[str],
    extra_host_tensor_names: list[str],
) -> list[PhasePlan]:
    """One :class:`PhasePlan` per ``hl.barrier()``-separated phase, in order.

    A phase's ``input_names`` is every host tensor name its graphs reference
    that is a declared parameter, a discovered extra host tensor, or an
    earlier phase's output name (bound by the driver after that phase runs).
    A phase's ``outputs`` reuses :class:`OutputTensorResolver`'s precedence
    rules, scoped to just this phase's own graphs (root + nested loop
    bodies) instead of the whole kernel's.
    """
    from .output_resolver import OutputTensorResolver

    known_names: set[str] = set(declared_param_names) | set(extra_host_tensor_names)
    resolver = OutputTensorResolver(hf)
    plans: list[PhasePlan] = []
    for phase_index, phase in enumerate(hf.device_ir.phases):
        phase_graphs = iter_phase_graphs(hf, phase.roots)
        outputs = resolver.resolve_all_in_graphs(phase_graphs)
        # A store's destination is itself referenced via `_host_tensor(name)`
        # (to get the tensor to store into), so it shows up in the raw scan
        # below even though the phase only *writes* it, never reads its
        # existing value -- exclude a phase's own outputs from its inputs.
        own_output_names = {
            name
            for _, tensor in outputs
            if (name := resolve_host_variable_name(hf, tensor)) is not None
        }

        referenced = _phase_host_tensor_names(hf, phase.roots)
        input_names = [
            name
            for name in referenced
            if name in known_names and name not in own_output_names
        ]

        for _, tensor in outputs:
            resolved_name = resolve_host_variable_name(hf, tensor)
            if resolved_name is not None:
                known_names.add(resolved_name)

        plans.append(
            PhasePlan(
                phase_index=phase_index,
                root_ids=list(phase.roots),
                input_names=input_names,
                outputs=outputs,
            )
        )
    return plans
