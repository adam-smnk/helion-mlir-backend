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
import uuid

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
    block_hint_to_id: dict[int, int] | None = None,
    block_symint_to_id: dict[int, int] | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
    arg_position_overrides: dict[int, dict[int, torch.Tensor]] | None = None,
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
    ir_text, name_map = _batch_import_and_lower(
        aten_nodes,
        block_id_to_size or {},
        block_hint_to_id or {},
        block_symint_to_id or {},
        block_id_to_upper_bound or {},
        arg_position_overrides or {},
    )

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
    block_hint_to_id: dict[int, int] | None = None,
    block_symint_to_id: dict[int, int] | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
    arg_position_overrides: dict[int, dict[int, torch.Tensor]] | None = None,
) -> tuple[str, dict[int, str]]:
    """Build and lower all ATen subgraphs in one torch-mlir pipeline pass.

    Each ATen node is materialized as its own helper function.

    Note: We intentionally do not deduplicate helpers across nodes because
    jagged/tiled kernels can carry subtly different shape bounds at different
    call sites even for semantically similar ops. Sharing helpers can then
    produce invalid func.call operand type mismatches during MLIR inlining.
    """
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

    for next_func_idx, node in enumerate(aten_nodes):
        # Include node identity and a nonce so repeated preprocess passes can
        # add variants without symbol collisions.
        func_name = f"_aten_{next_func_idx}_{id(node)}_{uuid.uuid4().hex[:8]}"
        try:
            graph, _ = _build_aten_subgraph(
                node,
                block_id_to_size,
                block_hint_to_id or {},
                block_symint_to_id or {},
                block_id_to_upper_bound or {},
                (arg_position_overrides or {}).get(id(node)),
            )
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
    block_hint_to_id: dict[int, int] | None = None,
    block_symint_to_id: dict[int, int] | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
    arg_position_override: dict[int, torch.Tensor] | None = None,
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
    concrete_tensor_args: dict[torch.fx.Node, torch.Tensor] = {}
    tensor_positions: list[int] = []

    for i, arg in enumerate(node_args):
        if isinstance(arg, torch.fx.Node):
            concrete_val = None
            if arg_position_override is not None and i in arg_position_override:
                concrete_val = arg_position_override[i]
            else:
                concrete_val = _fake_tensor_from_node_meta(
                    arg,
                    block_id_to_size or {},
                    block_hint_to_id or {},
                    block_symint_to_id or {},
                    block_id_to_upper_bound or {},
                )
            if concrete_val is not None:
                # Keep helper placeholders bounded by traced tensor metadata
                # so helper signatures match boundary tiles (e.g. 8x16 vs 16x16).
                bound_shape: list[int] | None = None
                tmeta = arg.meta.get("tensor_meta")
                if tmeta is not None:
                    tm_shape = getattr(tmeta, "shape", None)
                    if tm_shape is not None:
                        bound_shape = _resolve_dims(
                            tm_shape,
                            block_id_to_size or {},
                            block_hint_to_id or {},
                            block_symint_to_id or {},
                            block_id_to_upper_bound or {},
                        )
                if bound_shape is None:
                    arg_val = arg.meta.get("val")
                    if isinstance(arg_val, torch.Tensor):
                        bound_shape = _resolve_shape(
                            arg_val,
                            block_id_to_size or {},
                            block_hint_to_id or {},
                            block_symint_to_id or {},
                            block_id_to_upper_bound or {},
                        )
                if bound_shape is not None and len(bound_shape) == len(
                    concrete_val.shape
                ):
                    clipped = [
                        min(int(concrete_val.shape[d]), int(bound_shape[d]))
                        for d in range(len(bound_shape))
                    ]
                    if tuple(clipped) != tuple(int(s) for s in concrete_val.shape):
                        concrete_val = torch.zeros(clipped, dtype=concrete_val.dtype)

                ph = g.placeholder(f"arg{i}")
                ph.meta["val"] = concrete_val
                ph.meta["tensor_meta"] = _tensor_meta(concrete_val)
                placeholder_map[arg] = ph
                concrete_tensor_args[arg] = concrete_val
                tensor_positions.append(i)

    # Some traced elementwise ATen nodes can carry stale tensor metadata on one
    # operand while another operand is already tile-bounded. Normalize all
    # tensor placeholders for this node to a common bounded shape.
    target_name = str(node.target)
    if len(concrete_tensor_args) >= 2 and (
        "aten.add.Tensor" in target_name
        or "aten.mul.Tensor" in target_name
        or "aten.sub.Tensor" in target_name
        or "aten.div.Tensor" in target_name
    ):
        target_shape = _compute_broadcast_target_shape(
            [t for t in concrete_tensor_args.values() if isinstance(t, torch.Tensor)]
        )
        if target_shape is None:
            target_shape = _compute_conservative_common_shape(
                [
                    t
                    for t in concrete_tensor_args.values()
                    if isinstance(t, torch.Tensor)
                ]
            )
        if target_shape is not None:
            for fx_node, t in list(concrete_tensor_args.items()):
                if tuple(int(s) for s in t.shape) != tuple(target_shape):
                    broadcasted = torch.zeros(target_shape, dtype=t.dtype)
                    concrete_tensor_args[fx_node] = broadcasted
                    ph = placeholder_map.get(fx_node)
                    if ph is not None:
                        ph.meta["val"] = broadcasted
                        ph.meta["tensor_meta"] = _tensor_meta(broadcasted)

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

    # Prefer an eval-based concrete result shape when possible. FX result
    # metadata for nested tile loops can be stale/ambiguous for SymInt dims.
    concrete_result = _try_evaluate_aten_result(node, node_args, concrete_tensor_args)

    if concrete_result is None:
        result_val = node.meta.get("val")
        if isinstance(result_val, torch.Tensor):
            concrete_result = torch.zeros(
                _resolve_shape(
                    result_val,
                    block_id_to_size or {},
                    block_hint_to_id or {},
                    block_symint_to_id or {},
                    block_id_to_upper_bound or {},
                ),
                dtype=result_val.dtype,
            )
        elif isinstance(result_val, (list, tuple)):
            for rv in result_val:
                if isinstance(rv, torch.Tensor):
                    concrete_result = torch.zeros(
                        _resolve_shape(
                            rv,
                            block_id_to_size or {},
                            block_hint_to_id or {},
                            block_symint_to_id or {},
                            block_id_to_upper_bound or {},
                        ),
                        dtype=rv.dtype,
                    )
                    break
        else:
            concrete_result = _fake_tensor_from_node_meta(
                node,
                block_id_to_size or {},
                block_hint_to_id or {},
                block_symint_to_id or {},
                block_id_to_upper_bound or {},
            )

    if concrete_result is not None:
        aten_node.meta["val"] = concrete_result
        aten_node.meta["tensor_meta"] = _tensor_meta(concrete_result)

    g.output((aten_node,))
    return g, tensor_positions


def _try_evaluate_aten_result(
    node: torch.fx.Node,
    normalized_args: tuple[object, ...],
    concrete_tensor_args: dict[torch.fx.Node, torch.Tensor],
) -> torch.Tensor | None:
    """Attempt to evaluate one ATen op on fake concrete inputs for shape inference."""
    eval_args: list[object] = []
    for arg in normalized_args:
        if isinstance(arg, torch.fx.Node):
            if arg in concrete_tensor_args:
                eval_args.append(concrete_tensor_args[arg])
            else:
                literal = _resolve_fx_literal(arg)
                if literal is _MISSING_LITERAL:
                    return None
                eval_args.append(literal)
        else:
            eval_args.append(arg)

    eval_kwargs: dict[str, object] = {}
    for k, v in node.kwargs.items():
        if isinstance(v, torch.fx.Node):
            if v in concrete_tensor_args:
                eval_kwargs[k] = concrete_tensor_args[v]
            else:
                literal = _resolve_fx_literal(v)
                if literal is _MISSING_LITERAL:
                    return None
                eval_kwargs[k] = literal
        else:
            eval_kwargs[k] = v

    try:
        with torch.no_grad():
            out = node.target(*eval_args, **eval_kwargs)
    except Exception:
        return None

    if isinstance(out, torch.Tensor):
        return torch.zeros(tuple(int(d) for d in out.shape), dtype=out.dtype)

    if isinstance(out, (list, tuple)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return torch.zeros(tuple(int(d) for d in v.shape), dtype=v.dtype)

    return None


def _aten_node_fingerprint(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
) -> tuple[object, ...] | None:
    """Return a canonical key for ATen-node semantic deduplication.

    Returns ``None`` when we cannot safely canonicalize all inputs; callers
    should then skip deduplication for that node.
    """
    target_name = str(node.target)
    norm_args = _normalize_aten_args(node)

    arg_keys: list[object] = []
    for arg in norm_args:
        key = _fingerprint_arg(arg, block_id_to_size)
        if key is None:
            return None
        arg_keys.append(key)

    kwarg_keys: list[tuple[str, object]] = []
    for key, val in sorted(node.kwargs.items()):
        val_key = _fingerprint_arg(val, block_id_to_size)
        if val_key is None:
            return None
        kwarg_keys.append((key, val_key))

    result_sig = _tensor_signature_from_meta(node, block_id_to_size)

    return (
        "target",
        target_name,
        "args",
        tuple(arg_keys),
        "kwargs",
        tuple(kwarg_keys),
        "result",
        result_sig,
    )


def _fingerprint_arg(
    arg: object,
    block_id_to_size: dict[int, int],
) -> object | None:
    """Canonicalize one ATen argument for deduplication."""
    if isinstance(arg, torch.fx.Node):
        if _node_returns_tensor(arg):
            sig = _tensor_signature_from_meta(arg, block_id_to_size)
            if sig is None:
                return None
            return ("tensor", sig)

        literal = _resolve_fx_literal(arg)
        if literal is _MISSING_LITERAL:
            return None
        lit_key = _fingerprint_literal(literal)
        if lit_key is None:
            return None
        return ("literal", lit_key)

    lit_key = _fingerprint_literal(arg)
    if lit_key is None:
        return None
    return ("literal", lit_key)


def _tensor_signature_from_meta(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
) -> tuple[tuple[int, ...], str] | None:
    """Return a stable ``(shape, dtype)`` signature for a tensor FX node."""
    fake = _fake_tensor_from_node_meta(node, block_id_to_size)
    if fake is not None:
        return (tuple(int(d) for d in fake.shape), str(fake.dtype))

    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        shape = tuple(_resolve_shape(val, block_id_to_size))
        return (shape, str(val.dtype))

    tmeta = node.meta.get("tensor_meta")
    if tmeta is None:
        return None

    dtype = getattr(tmeta, "dtype", None)
    shape = getattr(tmeta, "shape", None)
    if dtype is None or shape is None:
        return None

    return (tuple(_resolve_dims(shape, block_id_to_size)), str(dtype))


def _fingerprint_literal(value: object) -> object | None:
    """Return a stable literal fingerprint or ``None`` if unsupported."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.device):
        return ("device", str(value))
    if isinstance(value, slice):
        start = _fingerprint_literal(value.start)
        stop = _fingerprint_literal(value.stop)
        step = _fingerprint_literal(value.step)
        if start is None or stop is None or step is None:
            return None
        return ("slice", start, stop, step)

    if isinstance(value, tuple):
        parts: list[object] = []
        for part in value:
            fp = _fingerprint_literal(part)
            if fp is None:
                return None
            parts.append(fp)
        return ("tuple", tuple(parts))

    if isinstance(value, list):
        parts = []
        for part in value:
            fp = _fingerprint_literal(part)
            if fp is None:
                return None
            parts.append(fp)
        return ("list", tuple(parts))

    return None


def _compute_broadcast_target_shape(tensors: list[torch.Tensor]) -> list[int] | None:
    """Return the common broadcast shape for tensors, or None if incompatible."""
    if not tensors:
        return None

    max_rank = max(len(t.shape) for t in tensors)
    result_rev: list[int] = []

    for rev_dim in range(max_rank):
        sizes = []
        for t in tensors:
            idx = len(t.shape) - 1 - rev_dim
            sizes.append(int(t.shape[idx]) if idx >= 0 else 1)

        dim_size = max(sizes)
        if any(s not in (1, dim_size) for s in sizes):
            return None
        result_rev.append(dim_size)

    return list(reversed(result_rev))


def _compute_conservative_common_shape(
    tensors: list[torch.Tensor],
) -> list[int] | None:
    """Return a conservative shape by taking per-dimension minimum non-one sizes.

    This is a fallback for stale tile metadata where strict broadcast checks can
    fail even though runtime tile extents are compatible.
    """
    if not tensors:
        return None

    max_rank = max(len(t.shape) for t in tensors)
    result_rev: list[int] = []

    for rev_dim in range(max_rank):
        sizes = []
        for t in tensors:
            idx = len(t.shape) - 1 - rev_dim
            sizes.append(int(t.shape[idx]) if idx >= 0 else 1)

        non_ones = [s for s in sizes if s != 1]
        if non_ones:
            result_rev.append(min(non_ones))
        else:
            result_rev.append(1)

    return list(reversed(result_rev))


def _resolve_shape(
    t: torch.Tensor,
    block_id_to_size: dict[int, int],
    block_hint_to_id: dict[int, int] | None = None,
    block_symint_to_id: dict[int, int] | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
) -> list[int]:
    """Return the concrete shape of *t*, resolving SymInt dims via *block_id_to_size*."""
    import sympy

    result = []
    for d in t.shape:
        if isinstance(d, torch.SymInt):
            hint_val: int | None = None
            try:
                hint_val = int(d)
            except Exception:
                hint_val = None

            sym = str(d)
            if sym.startswith("u") and sym[1:].isdigit():
                block_id = int(sym[1:])
                if block_id in block_id_to_size:
                    mapped = block_id_to_size[block_id]
                    ub = (
                        block_id_to_upper_bound.get(block_id)
                        if block_id_to_upper_bound is not None
                        else None
                    )
                    if ub is not None and ub > 0:
                        mapped = min(mapped, ub)
                    if hint_val is not None and hint_val > 0:
                        mapped = min(mapped, hint_val)
                    result.append(mapped)
                    continue
            expr = getattr(getattr(d, "node", None), "expr", None)
            if isinstance(expr, sympy.Symbol):
                sym2 = str(expr)
                if sym2.startswith("u") and sym2[1:].isdigit():
                    block_id = int(sym2[1:])
                    if block_id in block_id_to_size:
                        mapped = block_id_to_size[block_id]
                        ub = (
                            block_id_to_upper_bound.get(block_id)
                            if block_id_to_upper_bound is not None
                            else None
                        )
                        if ub is not None and ub > 0:
                            mapped = min(mapped, ub)
                        if hint_val is not None and hint_val > 0:
                            mapped = min(mapped, hint_val)
                        result.append(mapped)
                        continue
            # Fallback: identity-based mapping when symbolic names are not usable.
            if block_symint_to_id:
                expr = getattr(getattr(d, "node", None), "expr", None)
                if isinstance(expr, sympy.Symbol) and id(expr) in block_symint_to_id:
                    block_id = block_symint_to_id[id(expr)]
                    mapped = block_id_to_size[block_id]
                    ub = (
                        block_id_to_upper_bound.get(block_id)
                        if block_id_to_upper_bound is not None
                        else None
                    )
                    if ub is not None and ub > 0:
                        mapped = min(mapped, ub)
                    if hint_val is not None and hint_val > 0:
                        mapped = min(mapped, hint_val)
                    result.append(mapped)
                    continue
        result.append(int(d))
    return result


def _tensor_meta(t: torch.Tensor) -> object:
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


def _resolve_fx_literal(node: torch.fx.Node) -> object:
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
    block_hint_to_id: dict[int, int] | None = None,
    block_symint_to_id: dict[int, int] | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
) -> torch.Tensor | None:
    """Construct a concrete fake tensor from ``node.meta`` when possible."""
    # Handle helion load nodes explicitly: derive tile extents from source
    # tensor shape and index block sizes, clipped by source bounds.
    if (
        node.op == "call_function"
        and getattr(node.target, "__name__", "") == "load"
        and len(node.args) >= 2
    ):
        tensor_arg = node.args[0]
        index_nodes = node.args[1]
        if isinstance(tensor_arg, torch.fx.Node) and isinstance(
            index_nodes, (list, tuple)
        ):
            source = _fake_tensor_from_node_meta(
                tensor_arg,
                block_id_to_size,
                block_hint_to_id,
                block_symint_to_id,
                block_id_to_upper_bound,
            )
            if source is not None:
                import sympy

                # Advanced indexing with a single tensor index should preserve
                # the full rank of the index tensor (e.g. x_flat[flat_indices]).
                if len(index_nodes) == 1 and isinstance(index_nodes[0], torch.fx.Node):
                    idx_fake = _fake_tensor_from_node_meta(
                        index_nodes[0],
                        block_id_to_size,
                        block_hint_to_id,
                        block_symint_to_id,
                        block_id_to_upper_bound,
                    )
                    if isinstance(idx_fake, torch.Tensor):
                        return torch.zeros(
                            _resolve_shape(
                                idx_fake,
                                block_id_to_size,
                                block_hint_to_id,
                                block_symint_to_id,
                                block_id_to_upper_bound,
                            ),
                            dtype=source.dtype,
                        )

                src_shape = [int(d) for d in source.shape]
                out_shape: list[int] = []

                for dim, idx_node in enumerate(index_nodes):
                    if dim >= len(src_shape):
                        break

                    extent = src_shape[dim]
                    block_id: int | None = None

                    if isinstance(idx_node, torch.fx.Node):
                        tname = getattr(idx_node.target, "__name__", "")
                        if tname == "_get_symnode" and idx_node.args:
                            key = idx_node.args[0]
                            if isinstance(key, str) and key.startswith("block_size_"):
                                try:
                                    block_id = int(key.split("_")[-1])
                                except Exception:
                                    block_id = None

                        idx_val = idx_node.meta.get("val")
                        if isinstance(idx_val, torch.SymInt):
                            sym = str(idx_val)
                            if sym.startswith("u") and sym[1:].isdigit():
                                block_id = int(sym[1:])
                            if block_id is None:
                                expr = getattr(
                                    getattr(idx_val, "node", None), "expr", None
                                )
                                if isinstance(expr, sympy.Symbol):
                                    sym2 = str(expr)
                                    if sym2.startswith("u") and sym2[1:].isdigit():
                                        block_id = int(sym2[1:])
                                    elif (
                                        block_symint_to_id is not None
                                        and id(expr) in block_symint_to_id
                                    ):
                                        block_id = block_symint_to_id[id(expr)]
                    if block_id is not None and block_id in block_id_to_size:
                        mapped = int(block_id_to_size[block_id])
                        ub = (
                            block_id_to_upper_bound.get(block_id)
                            if block_id_to_upper_bound is not None
                            else None
                        )
                        if ub is not None and ub > 0:
                            mapped = min(mapped, int(ub))
                        extent = min(extent, mapped)
                    out_shape.append(extent)

                if out_shape:
                    return torch.zeros(out_shape, dtype=source.dtype)

    # Prefer evaluating tensor-producing call_function nodes from their
    # concrete input tensors. This avoids stale/ambiguous symbolic metadata in
    # nested tile loops (e.g. matmul feeding add in reduction bodies).
    if node.op == "call_function":
        eval_args: list[object] = []
        can_eval = True
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                concrete = _fake_tensor_from_node_meta(
                    arg,
                    block_id_to_size,
                    block_hint_to_id,
                    block_symint_to_id,
                    block_id_to_upper_bound,
                )
                if concrete is not None:
                    eval_args.append(concrete)
                    continue
                literal = _resolve_fx_literal(arg)
                if literal is not _MISSING_LITERAL:
                    eval_args.append(literal)
                    continue
                can_eval = False
                break
            eval_args.append(arg)

        eval_kwargs: dict[str, object] = {}
        if can_eval:
            for k, v in node.kwargs.items():
                if isinstance(v, torch.fx.Node):
                    concrete = _fake_tensor_from_node_meta(
                        v,
                        block_id_to_size,
                        block_hint_to_id,
                        block_symint_to_id,
                        block_id_to_upper_bound,
                    )
                    if concrete is not None:
                        eval_kwargs[k] = concrete
                        continue
                    literal = _resolve_fx_literal(v)
                    if literal is not _MISSING_LITERAL:
                        eval_kwargs[k] = literal
                        continue
                    can_eval = False
                    break
                eval_kwargs[k] = v

        if can_eval:
            target_name = str(node.target)
            if (
                "aten.add.Tensor" in target_name
                or "aten.mul.Tensor" in target_name
                or "aten.sub.Tensor" in target_name
                or "aten.div.Tensor" in target_name
            ):
                tensor_arg_idxs = [
                    i for i, a in enumerate(eval_args) if isinstance(a, torch.Tensor)
                ]
                if len(tensor_arg_idxs) >= 2:
                    target_shape = _compute_broadcast_target_shape(
                        [
                            eval_args[i]
                            for i in tensor_arg_idxs
                            if isinstance(eval_args[i], torch.Tensor)
                        ]
                    )
                    if target_shape is None:
                        target_shape = _compute_conservative_common_shape(
                            [
                                eval_args[i]
                                for i in tensor_arg_idxs
                                if isinstance(eval_args[i], torch.Tensor)
                            ]
                        )
                    if target_shape is not None:
                        for i in tensor_arg_idxs:
                            t = eval_args[i]
                            if isinstance(t, torch.Tensor) and tuple(
                                int(s) for s in t.shape
                            ) != tuple(target_shape):
                                eval_args[i] = torch.zeros(target_shape, dtype=t.dtype)

        if can_eval:
            try:
                with torch.no_grad():
                    out = node.target(*eval_args, **eval_kwargs)
            except Exception:
                out = None

            if isinstance(out, torch.Tensor):
                return torch.zeros(tuple(int(d) for d in out.shape), dtype=out.dtype)
            if isinstance(out, (list, tuple)):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        return torch.zeros(
                            tuple(int(d) for d in item.shape), dtype=item.dtype
                        )

    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        shape = _resolve_shape(
            val,
            block_id_to_size,
            block_hint_to_id,
            block_symint_to_id,
            block_id_to_upper_bound,
        )
        return torch.zeros(shape, dtype=val.dtype)

    tmeta = node.meta.get("tensor_meta")
    if tmeta is None:
        return None

    dtype = getattr(tmeta, "dtype", None)
    shape = getattr(tmeta, "shape", None)
    if dtype is None or shape is None:
        return None

    concrete_shape = _resolve_dims(
        shape,
        block_id_to_size,
        block_hint_to_id,
        block_symint_to_id,
        block_id_to_upper_bound,
    )
    return torch.zeros(concrete_shape, dtype=dtype)


def _resolve_dims(
    dims: tuple[object, ...] | list[object],
    block_id_to_size: dict[int, int],
    block_hint_to_id: dict[int, int] | None = None,
    block_symint_to_id: dict[int, int] | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
) -> list[int]:
    """Resolve a sequence of dims (possibly SymInt) to concrete integer sizes."""
    import sympy

    result = []
    for d in dims:
        if isinstance(d, torch.SymInt):
            hint_val: int | None = None
            try:
                hint_val = int(d)
            except Exception:
                hint_val = None

            sym = str(d)
            if sym.startswith("u") and sym[1:].isdigit():
                block_id = int(sym[1:])
                if block_id in block_id_to_size:
                    mapped = block_id_to_size[block_id]
                    ub = (
                        block_id_to_upper_bound.get(block_id)
                        if block_id_to_upper_bound is not None
                        else None
                    )
                    if ub is not None and ub > 0:
                        mapped = min(mapped, ub)
                    if hint_val is not None and hint_val > 0:
                        mapped = min(mapped, hint_val)
                    result.append(mapped)
                    continue
            expr = getattr(getattr(d, "node", None), "expr", None)
            if isinstance(expr, sympy.Symbol):
                sym2 = str(expr)
                if sym2.startswith("u") and sym2[1:].isdigit():
                    block_id = int(sym2[1:])
                    if block_id in block_id_to_size:
                        mapped = block_id_to_size[block_id]
                        ub = (
                            block_id_to_upper_bound.get(block_id)
                            if block_id_to_upper_bound is not None
                            else None
                        )
                        if ub is not None and ub > 0:
                            mapped = min(mapped, ub)
                        if hint_val is not None and hint_val > 0:
                            mapped = min(mapped, hint_val)
                        result.append(mapped)
                        continue
            if block_symint_to_id:
                expr = getattr(getattr(d, "node", None), "expr", None)
                if isinstance(expr, sympy.Symbol) and id(expr) in block_symint_to_id:
                    block_id = block_symint_to_id[id(expr)]
                    mapped = block_id_to_size[block_id]
                    ub = (
                        block_id_to_upper_bound.get(block_id)
                        if block_id_to_upper_bound is not None
                        else None
                    )
                    if ub is not None and ub > 0:
                        mapped = min(mapped, ub)
                    if hint_val is not None and hint_val > 0:
                        mapped = min(mapped, hint_val)
                    result.append(mapped)
                    continue
        result.append(int(d))
    return result


def _infer_node_dtype(node: torch.fx.Node) -> torch.dtype | None:
    """Infer a dtype from FX node metadata if possible."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype

    tmeta = node.meta.get("tensor_meta")
    if tmeta is not None:
        return getattr(tmeta, "dtype", None)

    return None


def _normalize_aten_args(node: torch.fx.Node) -> tuple[object, ...]:
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


def _sym_name(op: object) -> str | None:
    """Return the ``sym_name`` string of an op, or None if it has none."""
    try:
        import mlir.ir as ir

        return ir.StringAttr(op.attributes["sym_name"]).value
    except (KeyError, Exception):
        return None


def _find_func(module: ir.Module, func_name: str) -> ir.Operation | None:
    """Return the op with ``sym_name == func_name``, or None."""
    for op in module.body.operations:
        if _sym_name(op) == func_name:
            return op
    return None
