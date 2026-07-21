"""Bridge: lower ATen FX nodes to linalg-on-tensors via torch-mlir.

Architecture (pre-pass, single pipeline run)
--------------------------------------------
Before codegen begins, :func:`preprocess_aten_nodes` is called with every
ATen ``call_function`` node found in the device IR.  It:

1. Builds a minimal ``torch.fx.Graph`` for each node (placeholder nodes for
   tensor inputs; scalar args are embedded as literals).
2. Imports all graphs into **one** ``torch_mlir.ir`` module using
   ``FxImporter.import_stateless_graph``, giving each function a unique name.
3. Runs the two-stage torch-mlir lowering pipeline **once** on that module:
   - ``torchdynamo-export-to-torch-backend-pipeline``
   - ``torch-backend-to-linalg-on-tensors-backend-pipeline``
4. Serialises the resulting linalg module to text and re-parses it into the
   caller's ``mlir.ir.Context``.
5. Clones every ``func.func`` into the main module's body **at module level**
   (not nested inside any ``scf.for`` / ``scf.forall``), making them valid
   top-level symbol-table entries.

During codegen each ATen node is handled with a simple ``func.call`` to its
pre-built helper — no per-node serialisation, no pipeline overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.fx

from .errors import NodeLoweringError

if TYPE_CHECKING:
    import mlir.ir as ir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_aten_op(node: torch.fx.Node) -> bool:
    """Return True if *node* is a standard ATen op that torch-mlir can lower.

    Requirements:
    - It is a ``call_function`` with a ``TorchOpOverload`` target (standard ATen).
    - Its result is a ``torch.Tensor`` (excludes shape-query ops like
      ``aten.sym_size.int`` whose result is a ``torch.SymInt``).
    """
    from torch._ops import OpOverload as TorchOpOverload

    return (
        node.op == "call_function"
        and isinstance(node.target, TorchOpOverload)
        and _node_returns_tensor(node)
    )


def collect_tensor_input_positions(node: torch.fx.Node) -> list[int]:
    """Return the argument positions in *node.args* that are tensor-valued."""
    positions = []
    for i, arg in enumerate(_normalize_aten_args(node)):
        if isinstance(arg, torch.fx.Node) and _node_returns_tensor(arg):
            positions.append(i)
    return positions


def normalized_aten_args(node: torch.fx.Node) -> tuple:
    """Return ATen args after applying known normalization rules."""
    return _normalize_aten_args(node)


def preprocess_aten_nodes(
    aten_nodes: list[torch.fx.Node],
    mlir_module: ir.Module,
    block_id_to_size: dict[int, int] | None = None,
) -> dict[int, tuple[str, list[ir.Type]]]:
    """Lower *all* ATen nodes in a single torch-mlir pipeline pass.

    Parameters
    ----------
    aten_nodes:
        Every ATen ``call_function`` node found in the device IR, in any order.
    mlir_module:
        The main ``mlir.ir.Module`` being built.  The lowered helper
        ``func.func`` operations are cloned into this module's body at module
        top level (i.e. outside any other function or region).
    block_id_to_size:
        Mapping from helion block_id to concrete integer size (from
        ``CompileEnvironment.block_sizes``).  Used to resolve SymInt tensor
        dimensions to their correct concrete values so that the helper
        function signatures use static shapes that match the extract_slice
        results produced during codegen.  If *None*, the SymInt hint value is
        used as a fallback (may produce dynamic-sized helpers).

    Returns
    -------
    dict mapping ``id(node)`` → ``(func_name, return_types)`` for every node
    in *aten_nodes*.
    """
    import mlir.ir as ir

    if not aten_nodes:
        return {}

    # --- Phase 1: build one torch_mlir module with all ATen subgraphs ------
    ir_text, name_map = _batch_import_and_lower(aten_nodes, block_id_to_size or {})

    # --- Phase 2: parse into our mlir.ir context ----------------------------
    helper_mod = ir.Module.parse(ir_text)

    # --- Phase 3: clone helpers into module.body at top level ---------------
    # Use an explicit InsertionPoint.at_block_begin so helpers are placed at
    # module scope regardless of what active InsertionPoint the caller holds.
    # A nested InsertionPoint context overrides the outer one, so this removes
    # the implicit coupling to the caller's build() context manager.
    # ``op.clone()`` with an active IP auto-inserts at that point.
    import mlir.ir as ir

    with ir.InsertionPoint.at_block_begin(mlir_module.body):
        for op in helper_mod.body.operations:
            sym_name = _sym_name(op)
            if sym_name and sym_name in name_map.values():
                cloned = op.clone()  # auto-inserted at at_block_begin
                cloned.attributes["sym_visibility"] = ir.StringAttr.get("private")

    # --- Phase 4: build return-type map -------------------------------------
    node_to_func: dict[int, tuple[str, list[ir.Type]]] = {}
    for node_id, func_name in name_map.items():
        func_op = _find_func(helper_mod, func_name)
        if func_op is None:
            log.warning("Helper '%s' not found in lowered module", func_name)
            continue
        ftype = ir.FunctionType(ir.TypeAttr(func_op.attributes["function_type"]).value)
        node_to_func[node_id] = (func_name, list(ftype.results))

    return node_to_func


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _batch_import_and_lower(
    aten_nodes: list[torch.fx.Node],
    block_id_to_size: dict[int, int],
) -> tuple[str, dict[int, str]]:
    """Build and lower all ATen subgraphs in one torch-mlir pipeline pass."""
    from torch_mlir.compiler_utils import OutputType
    from torch_mlir.compiler_utils import lower_mlir_module
    from torch_mlir.compiler_utils import run_pipeline_with_repro_report
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter
    import torch_mlir.ir as tm_ir

    tm_ctx = tm_ir.Context()
    torch_d.register_dialect(tm_ctx)
    imp = FxImporter(context=tm_ctx)

    name_map: dict[int, str] = {}

    for idx, node in enumerate(aten_nodes):
        func_name = f"_aten_{idx}"
        try:
            graph, _ = _build_aten_subgraph(node, block_id_to_size)
            imp.import_stateless_graph(graph, func_name=func_name)
            name_map[id(node)] = func_name
        except Exception as exc:
            arg_info = []
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    arg_info.append(
                        f"{a.name}:{a.op}:{a.target}:val={type(a.meta.get('val')).__name__ if a.meta.get('val') is not None else 'None'}:tm={a.meta.get('tensor_meta') is not None}"
                    )
                else:
                    arg_info.append(f"lit:{type(a).__name__}:{a}")
            kw_info = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.fx.Node):
                    kw_info[k] = (
                        f"{v.name}:{v.op}:{v.target}:val={type(v.meta.get('val')).__name__ if v.meta.get('val') is not None else 'None'}:tm={v.meta.get('tensor_meta') is not None}"
                    )
                else:
                    kw_info[k] = f"lit:{type(v).__name__}:{v}"
            log.warning(
                "Could not import ATen node '%s' (%s): %s | args=%s kwargs=%s",
                node.name,
                node.target,
                exc,
                arg_info,
                kw_info,
            )

    # Two-stage torch-mlir pipeline (single run for all subgraphs)
    run_pipeline_with_repro_report(
        imp.module,
        "builtin.module(func.func(torch-match-quantized-custom-ops),"
        " torchdynamo-export-to-torch-backend-pipeline{})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )
    lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, imp.module)

    ir_text = imp.module.operation.get_asm(binary=False, enable_debug_info=False)
    return ir_text, name_map


def _build_aten_subgraph(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int] | None = None,
) -> tuple[torch.fx.Graph, list[int]]:
    """Build a minimal ``torch.fx.Graph`` for a single ATen node.

    Tensor inputs become ``placeholder`` nodes with concrete static shapes.
    SymInt dimensions are resolved using *block_id_to_size* (when provided),
    falling back to ``int(dim)`` (the SymInt hint) otherwise.
    Scalar args are embedded as literals.
    """
    g = torch.fx.Graph()
    node_args = _normalize_aten_args(node)
    placeholder_map: dict[torch.fx.Node, torch.fx.Node] = {}
    tensor_positions: list[int] = []

    for i, arg in enumerate(node_args):
        if isinstance(arg, torch.fx.Node):
            concrete_val = _fake_tensor_from_node_meta(arg, block_id_to_size or {})
            if concrete_val is not None:
                ph = g.placeholder(f"arg{i}")
                ph.meta["val"] = concrete_val
                ph.meta["tensor_meta"] = _tensor_meta(concrete_val)
                placeholder_map[arg] = ph
                tensor_positions.append(i)

    new_args = []
    for arg in node_args:
        if isinstance(arg, torch.fx.Node):
            if arg in placeholder_map:
                new_args.append(placeholder_map[arg])
            else:
                literal = _resolve_fx_literal(arg)
                if literal is _MISSING_LITERAL:
                    raise NodeLoweringError(
                        f"Cannot build ATen helper for node '{node.name}' ({node.target}): "
                        f"unresolved non-tensor input node '{arg.name}' ({arg.target})"
                    )
                new_args.append(literal)
        else:
            new_args.append(arg)

    new_kwargs = {}
    for k, v in node.kwargs.items():
        if isinstance(v, torch.fx.Node):
            if v in placeholder_map:
                new_kwargs[k] = placeholder_map[v]
            else:
                literal = _resolve_fx_literal(v)
                if literal is _MISSING_LITERAL:
                    raise NodeLoweringError(
                        f"Cannot build ATen helper for node '{node.name}' ({node.target}): "
                        f"unresolved kwarg node '{v.name}' ({v.target}) for kwarg '{k}'"
                    )
                new_kwargs[k] = literal
        else:
            new_kwargs[k] = v

    aten_node = g.call_function(node.target, tuple(new_args), new_kwargs)

    result_val = node.meta.get("val")
    if isinstance(result_val, torch.Tensor):
        concrete_result = torch.zeros(
            _resolve_shape(result_val, block_id_to_size or {}),
            dtype=result_val.dtype,
        )
        aten_node.meta["val"] = concrete_result
        aten_node.meta["tensor_meta"] = _tensor_meta(concrete_result)
    elif isinstance(result_val, (list, tuple)):
        for rv in result_val:
            if isinstance(rv, torch.Tensor):
                concrete_rv = torch.zeros(
                    _resolve_shape(rv, block_id_to_size or {}),
                    dtype=rv.dtype,
                )
                aten_node.meta["val"] = concrete_rv
                aten_node.meta["tensor_meta"] = _tensor_meta(concrete_rv)
                break
    else:
        concrete_result = _fake_tensor_from_node_meta(node, block_id_to_size or {})
        if concrete_result is not None:
            aten_node.meta["val"] = concrete_result
            aten_node.meta["tensor_meta"] = _tensor_meta(concrete_result)

    g.output((aten_node,))
    return g, tensor_positions


def _resolve_shape(t: torch.Tensor, block_id_to_size: dict[int, int]) -> list[int]:
    """Return the concrete shape of *t*, resolving SymInt dims via *block_id_to_size*.

    SymInt dimensions have string representations like ``"u0"``, ``"u1"``…
    where the number is the helion block_id.  We map that to the concrete
    block size so the helper function uses the same static shapes that
    ``_lower_load`` will produce for the corresponding ``tensor.extract_slice``.
    """
    result = []
    for d in t.shape:
        if isinstance(d, torch.SymInt):
            sym = str(d)  # e.g. "u0", "u1", "u2"
            if sym.startswith("u") and sym[1:].isdigit():
                block_id = int(sym[1:])
                if block_id in block_id_to_size:
                    result.append(block_id_to_size[block_id])
                    continue
        result.append(int(d))
    return result


def _tensor_meta(t: torch.Tensor):
    from torch.fx.passes.shape_prop import TensorMetadata

    return TensorMetadata(
        shape=t.shape,
        dtype=t.dtype,
        requires_grad=t.requires_grad,
        stride=t.stride() if not t.is_sparse else (),
        memory_format=None,
        is_quantized=t.is_quantized,
        qparams={},
    )


_MISSING_LITERAL = object()


def _node_returns_tensor(node: torch.fx.Node) -> bool:
    """Return True if node metadata indicates a tensor result."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return True
    return node.meta.get("tensor_meta") is not None


def _resolve_fx_literal(node: torch.fx.Node):
    """Resolve a non-tensor FX node to a Python literal for subgraph import.

    Returns ``_MISSING_LITERAL`` when unresolved.
    """
    val = node.meta.get("val", _MISSING_LITERAL)
    if val is not _MISSING_LITERAL and val is not None:
        return val

    # Common case in dtype/cast-heavy graphs: dtype flows through a small FX
    # helper node (e.g. getattr(tensor, "dtype")) with val=None metadata.
    if node.op == "call_function" and len(node.args) >= 2:
        target_name = getattr(node.target, "__name__", "")
        attr_name = node.args[1]
        if target_name == "getattr" and attr_name == "dtype":
            src = node.args[0]
            if isinstance(src, torch.fx.Node):
                inferred = _infer_node_dtype(src)
                if inferred is not None:
                    return inferred

    inferred = _infer_node_dtype(node)
    if inferred is not None:
        return inferred

    return _MISSING_LITERAL


def _fake_tensor_from_node_meta(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
) -> torch.Tensor | None:
    """Construct a concrete fake tensor from ``node.meta`` when possible."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        shape = _resolve_shape(val, block_id_to_size)
        return torch.zeros(shape, dtype=val.dtype)

    tmeta = node.meta.get("tensor_meta")
    if tmeta is None:
        return None

    dtype = getattr(tmeta, "dtype", None)
    shape = getattr(tmeta, "shape", None)
    if dtype is None or shape is None:
        return None

    concrete_shape = _resolve_dims(shape, block_id_to_size)
    return torch.zeros(concrete_shape, dtype=dtype)


def _resolve_dims(dims, block_id_to_size: dict[int, int]) -> list[int]:
    """Resolve a sequence of dims (possibly SymInt) to concrete integer sizes."""
    result = []
    for d in dims:
        if isinstance(d, torch.SymInt):
            sym = str(d)
            if sym.startswith("u") and sym[1:].isdigit():
                block_id = int(sym[1:])
                if block_id in block_id_to_size:
                    result.append(block_id_to_size[block_id])
                    continue
        result.append(int(d))
    return result


def _infer_node_dtype(node: torch.fx.Node):
    """Infer a dtype from FX node metadata if possible."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype

    tmeta = node.meta.get("tensor_meta")
    if tmeta is not None:
        return getattr(tmeta, "dtype", None)

    return None


def _normalize_aten_args(node: torch.fx.Node):
    """Return sanitized ATen args for known FX tracing quirks.

    In some cast-heavy patterns, FX can encode ``aten.mul.Tensor`` as
    ``(tensor_node, None)`` when the intended operation is self-multiplication.
    Normalize that to ``(tensor_node, tensor_node)`` so torch-mlir import
    receives a valid Tensor/Tensor operand pair.
    """
    args = list(node.args)
    target_name = str(node.target)
    if (
        "aten.mul.Tensor" in target_name
        and len(args) == 2
        and args[1] is None
        and isinstance(args[0], torch.fx.Node)
    ):
        args[1] = args[0]
    return tuple(args)


def _sym_name(op) -> str | None:
    """Return the ``sym_name`` string of an op, or None if it has none."""
    try:
        import mlir.ir as ir

        return ir.StringAttr(op.attributes["sym_name"]).value
    except (KeyError, Exception):
        return None


def _find_func(module: ir.Module, func_name: str):
    """Return the op with ``sym_name == func_name``, or None."""
    for op in module.body.operations:
        if _sym_name(op) == func_name:
            return op
    return None
