"""Torch-MLIR batch import and lowering orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Callable
import uuid

import torch
import torch.fx

if TYPE_CHECKING:
    from collections.abc import Callable as CallableType

    from helion._compiler.compile_environment import CompileEnvironment

    AtenSubgraphBuilder = CallableType[..., tuple[torch.fx.Graph, list[int]]]

log = logging.getLogger(__name__)


def batch_import_and_lower(
    aten_nodes: list[torch.fx.Node],
    block_id_to_size: dict[int, int],
    build_subgraph: Callable[..., tuple[torch.fx.Graph, list[int]]],
    env: CompileEnvironment | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
    arg_position_overrides: dict[int, dict[int, torch.Tensor]] | None = None,
) -> tuple[str, dict[int, str]]:
    """Build and lower all ATen subgraphs in one torch-MLIR pipeline pass."""
    try:
        from torch_mlir.compiler_utils import OutputType
        from torch_mlir.compiler_utils import lower_mlir_module
        from torch_mlir.compiler_utils import run_pipeline_with_repro_report
        from torch_mlir.dialects import torch as torch_d
        from torch_mlir.extras.fx_importer import FxImporter
        import torch_mlir.ir as tm_ir
    except ImportError as exc:
        raise ImportError(
            "torch-mlir is required to lower ATen ops for the MLIR backend "
            "(every ATen op not covered by a hand-written manual lowering is "
            "compiled through it). Install it with the project's pinned "
            "dependency set (e.g. `uv sync`)."
        ) from exc

    context = tm_ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    name_map: dict[int, str] = {}

    for function_index, node in enumerate(aten_nodes):
        function_name = f"_aten_{function_index}_{id(node)}_{uuid.uuid4().hex[:8]}"
        try:
            graph, _ = build_subgraph(
                node,
                block_id_to_size,
                env,
                block_id_to_upper_bound or {},
                (arg_position_overrides or {}).get(id(node)),
            )
            importer.import_stateless_graph(graph, func_name=function_name)
            name_map[id(node)] = function_name
        except Exception as exc:
            # FxImporter/torch-mlir can fail in many ways for an unsupported
            # op; skip this node's helper and let codegen raise a clear
            # UnsupportedOperationError when it finds no pre-built entry.
            arg_info = [
                (
                    f"{arg.name}:{arg.op}:{arg.target}:"
                    f"val={type(arg.meta.get('val')).__name__ if arg.meta.get('val') is not None else 'None'}:"
                    f"tm={arg.meta.get('tensor_meta') is not None}"
                    if isinstance(arg, torch.fx.Node)
                    else f"lit:{type(arg).__name__}:{arg}"
                )
                for arg in node.args
            ]
            kw_info = {
                key: (
                    f"{value.name}:{value.op}:{value.target}:"
                    f"val={type(value.meta.get('val')).__name__ if value.meta.get('val') is not None else 'None'}:"
                    f"tm={value.meta.get('tensor_meta') is not None}"
                    if isinstance(value, torch.fx.Node)
                    else f"lit:{type(value).__name__}:{value}"
                )
                for key, value in node.kwargs.items()
            }
            log.warning(
                "Could not import ATen node '%s' (%s): %s | args=%s kwargs=%s",
                node.name,
                node.target,
                exc,
                arg_info,
                kw_info,
            )

    run_pipeline_with_repro_report(
        importer.module,
        "builtin.module(func.func(torch-match-quantized-custom-ops),"
        " torchdynamo-export-to-torch-backend-pipeline{})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )
    lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, importer.module)
    return (
        importer.module.operation.get_asm(binary=False, enable_debug_info=False),
        name_map,
    )
