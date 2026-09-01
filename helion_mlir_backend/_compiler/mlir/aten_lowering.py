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

from .support import NodeLoweringError
from .support import block_id_from_key

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
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


def _is_broadcasting_aten_target(target: object) -> bool:
    target_name = str(target)
    return any(
        operation in target_name
        for operation in (
            "aten.add.Tensor",
            "aten.mul.Tensor",
            "aten.sub.Tensor",
            "aten.div.Tensor",
        )
    )


def preprocess_aten_nodes(
    aten_nodes: list[torch.fx.Node],
    mlir_module: ir.Module,
    block_id_to_size: dict[int, int] | None = None,
    env: CompileEnvironment | None = None,
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
    from .aten_bridge import batch_import_and_lower

    ir_text, name_map = batch_import_and_lower(
        aten_nodes,
        block_id_to_size or {},
        _build_aten_subgraph,
        env,
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


def _build_aten_subgraph(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int] | None = None,
    env: CompileEnvironment | None = None,
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
                    env,
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
                            env,
                            block_id_to_upper_bound or {},
                        )
                if bound_shape is None:
                    arg_val = arg.meta.get("val")
                    if isinstance(arg_val, torch.Tensor):
                        bound_shape = _resolve_dims(
                            arg_val.shape,
                            block_id_to_size or {},
                            env,
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
    if len(concrete_tensor_args) >= 2 and _is_broadcasting_aten_target(node.target):
        target_shape = broadcast_target_shape(
            [t for t in concrete_tensor_args.values() if isinstance(t, torch.Tensor)]
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
        concrete_result = _result_tensor_from_node_meta(
            node,
            block_id_to_size or {},
            env,
            block_id_to_upper_bound or {},
        )

    if concrete_result is not None:
        aten_node.meta["val"] = concrete_result
        aten_node.meta["tensor_meta"] = _tensor_meta(concrete_result)

    g.output((aten_node,))
    return g, tensor_positions


def _result_tensor_from_node_meta(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
    env: CompileEnvironment | None,
    block_id_to_upper_bound: dict[int, int],
) -> torch.Tensor | None:
    """Construct a concrete result tensor from fallback FX metadata."""
    result_val = node.meta.get("val")
    if isinstance(result_val, torch.Tensor):
        return torch.zeros(
            _resolve_dims(
                result_val.shape,
                block_id_to_size,
                env,
                block_id_to_upper_bound,
            ),
            dtype=result_val.dtype,
        )
    if isinstance(result_val, (list, tuple)):
        for value in result_val:
            if isinstance(value, torch.Tensor):
                return torch.zeros(
                    _resolve_dims(
                        value.shape,
                        block_id_to_size,
                        env,
                        block_id_to_upper_bound,
                    ),
                    dtype=value.dtype,
                )
    return _fake_tensor_from_node_meta(
        node,
        block_id_to_size,
        env,
        block_id_to_upper_bound,
    )


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
        # node.target is an arbitrary user-selected ATen op; it can raise any
        # exception type depending on operand shapes/dtypes.
        return None

    if isinstance(out, torch.Tensor):
        return torch.zeros(tuple(int(d) for d in out.shape), dtype=out.dtype)

    if isinstance(out, (list, tuple)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return torch.zeros(tuple(int(d) for d in v.shape), dtype=v.dtype)

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


def broadcast_target_shape(tensors: list[torch.Tensor]) -> list[int] | None:
    """Return a broadcast shape with a fallback for stale metadata."""
    return _compute_broadcast_target_shape(
        tensors
    ) or _compute_conservative_common_shape(tensors)


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


def _is_scalar_load_index(index_node: object) -> bool:
    """Return whether a Helion load index denotes a scalar position.

    A scalar position (grid index, ``tile.begin``/``tile.end``/``tile.id``, or
    a literal int) drops that dimension from the loaded value's rank, same as
    Helion's own ``node.meta['val']`` and ``ctx.is_scalar_index_node`` convention
    used elsewhere. A tile index (``block_size_N`` key) or a full slice keeps
    the dimension.
    """
    if isinstance(index_node, int):
        return True
    if not isinstance(index_node, torch.fx.Node):
        return False
    if getattr(index_node.target, "__name__", "") == "_get_symnode" and index_node.args:
        key = index_node.args[0]
        return not (isinstance(key, str) and key.startswith("block_size_"))
    return False


def _fake_tensor_from_load_node(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
    env: CompileEnvironment | None,
    block_id_to_upper_bound: dict[int, int] | None,
) -> torch.Tensor | None:
    """Infer the bounded result shape for a Helion indexed load."""
    if (
        node.op != "call_function"
        or getattr(node.target, "__name__", "") != "load"
        or len(node.args) < 2
    ):
        return None

    tensor_arg = node.args[0]
    index_nodes = node.args[1]
    if not isinstance(tensor_arg, torch.fx.Node) or not isinstance(
        index_nodes, (list, tuple)
    ):
        return None

    source = _fake_tensor_from_node_meta(
        tensor_arg,
        block_id_to_size,
        env,
        block_id_to_upper_bound,
    )
    if source is None:
        return None

    if len(index_nodes) == 1 and isinstance(index_nodes[0], torch.fx.Node):
        idx_fake = _fake_tensor_from_node_meta(
            index_nodes[0],
            block_id_to_size,
            env,
            block_id_to_upper_bound,
        )
        if isinstance(idx_fake, torch.Tensor):
            return torch.zeros(
                _resolve_dims(
                    idx_fake.shape,
                    block_id_to_size,
                    env,
                    block_id_to_upper_bound,
                ),
                dtype=source.dtype,
            )

    src_shape = [int(d) for d in source.shape]
    out_shape: list[int] = []
    for dim, idx_node in enumerate(index_nodes):
        if dim >= len(src_shape):
            break

        if _is_scalar_load_index(idx_node):
            # Dropped from the result's rank, matching Helion's own
            # rank-reduction convention (the load's real node.meta['val']).
            continue

        extent = src_shape[dim]
        block_id = _block_id_from_load_index(idx_node, env)

        if block_id is not None and block_id in block_id_to_size:
            mapped = int(block_id_to_size[block_id])
            upper_bound = (
                block_id_to_upper_bound.get(block_id)
                if block_id_to_upper_bound is not None
                else None
            )
            if upper_bound is not None and upper_bound > 0:
                mapped = min(mapped, int(upper_bound))
            extent = min(extent, mapped)
        out_shape.append(extent)

    if out_shape:
        return torch.zeros(out_shape, dtype=source.dtype)
    return torch.zeros([], dtype=source.dtype)


def _evaluate_fake_aten_node(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
    env: CompileEnvironment | None,
    block_id_to_upper_bound: dict[int, int] | None,
) -> torch.Tensor | None:
    """Evaluate an ATen node on recursively constructed fake operands."""
    if node.op != "call_function":
        return None

    eval_args: list[object] = []
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            concrete = _fake_tensor_from_node_meta(
                arg,
                block_id_to_size,
                env,
                block_id_to_upper_bound,
            )
            if concrete is not None:
                eval_args.append(concrete)
                continue
            literal = _resolve_fx_literal(arg)
            if literal is _MISSING_LITERAL:
                return None
            eval_args.append(literal)
            continue
        eval_args.append(arg)

    eval_kwargs: dict[str, object] = {}
    for key, value in node.kwargs.items():
        if isinstance(value, torch.fx.Node):
            concrete = _fake_tensor_from_node_meta(
                value,
                block_id_to_size,
                env,
                block_id_to_upper_bound,
            )
            if concrete is not None:
                eval_kwargs[key] = concrete
                continue
            literal = _resolve_fx_literal(value)
            if literal is _MISSING_LITERAL:
                return None
            eval_kwargs[key] = literal
            continue
        eval_kwargs[key] = value

    if _is_broadcasting_aten_target(node.target):
        tensor_args = [arg for arg in eval_args if isinstance(arg, torch.Tensor)]
        if len(tensor_args) >= 2:
            target_shape = broadcast_target_shape(tensor_args)
            if target_shape is not None:
                eval_args = [
                    (
                        torch.zeros(target_shape, dtype=arg.dtype)
                        if isinstance(arg, torch.Tensor)
                        and tuple(int(size) for size in arg.shape)
                        != tuple(target_shape)
                        else arg
                    )
                    for arg in eval_args
                ]

    try:
        with torch.no_grad():
            result = node.target(*eval_args, **eval_kwargs)
    except Exception:
        # Same rationale as _try_evaluate_aten_result: arbitrary ATen op.
        return None

    if isinstance(result, torch.Tensor):
        return torch.zeros(tuple(int(dim) for dim in result.shape), dtype=result.dtype)
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, torch.Tensor):
                return torch.zeros(
                    tuple(int(dim) for dim in item.shape), dtype=item.dtype
                )
    return None


def _fake_tensor_from_node_meta(
    node: torch.fx.Node,
    block_id_to_size: dict[int, int],
    env: CompileEnvironment | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
) -> torch.Tensor | None:
    """Construct a concrete fake tensor from ``node.meta`` when possible."""
    load_result = _fake_tensor_from_load_node(
        node,
        block_id_to_size,
        env,
        block_id_to_upper_bound,
    )
    if load_result is not None:
        return load_result

    # Prefer evaluated fake operands before consulting symbolic metadata.
    evaluated = _evaluate_fake_aten_node(
        node,
        block_id_to_size,
        env,
        block_id_to_upper_bound,
    )
    if evaluated is not None:
        return evaluated

    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        shape = _resolve_dims(
            val.shape,
            block_id_to_size,
            env,
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
        env,
        block_id_to_upper_bound,
    )
    return torch.zeros(concrete_shape, dtype=dtype)


def _resolve_dims(
    dims: tuple[object, ...] | list[object],
    block_id_to_size: dict[int, int],
    env: CompileEnvironment | None = None,
    block_id_to_upper_bound: dict[int, int] | None = None,
) -> list[int]:
    """Resolve a sequence of dims (possibly SymInt) to concrete integer sizes."""
    result = []
    for d in dims:
        if isinstance(d, torch.SymInt):
            resolved = _resolve_symint_dim(
                d,
                block_id_to_size,
                env,
                block_id_to_upper_bound,
            )
            if resolved is not None:
                result.append(resolved)
                continue
        result.append(int(d))
    return result


def _block_id_from_load_index(
    index_node: object,
    env: CompileEnvironment | None,
) -> int | None:
    if not isinstance(index_node, torch.fx.Node):
        return None
    target_name = getattr(index_node.target, "__name__", "")
    if target_name == "_get_symnode" and index_node.args:
        block_id = block_id_from_key(index_node.args[0])
        if block_id is not None:
            return block_id

    value = index_node.meta.get("val")
    if not isinstance(value, torch.SymInt) or env is None:
        return None
    return env.get_block_id(value)


def _resolve_symint_dim(
    dimension: torch.SymInt,
    block_id_to_size: dict[int, int],
    env: CompileEnvironment | None,
    block_id_to_upper_bound: dict[int, int] | None,
) -> int | None:
    try:
        hint_value = int(dimension)
    except (TypeError, ValueError):
        hint_value = None

    block_id = env.get_block_id(dimension) if env is not None else None
    if block_id is not None and block_id in block_id_to_size:
        resolved = _clamp_by_upper_bound(
            block_id_to_size[block_id], block_id, block_id_to_upper_bound
        )
        if hint_value is not None and hint_value > 0:
            resolved = min(resolved, hint_value)
        return resolved
    return None


def _clamp_by_upper_bound(
    value: int,
    block_id: int,
    upper_bounds: dict[int, int] | None,
) -> int:
    if upper_bounds is None:
        return value
    upper_bound = upper_bounds.get(block_id)
    if upper_bound is None or upper_bound <= 0:
        return value
    return min(value, upper_bound)


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
