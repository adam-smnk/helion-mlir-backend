"""Generic ATen nodes as calls to torch-mlir helper functions, typed at the call site.

At each call site the operands' MLIR types (tensors, and runtime scalars as
``i64``/``f64``/``i1``) and the node's literal arguments form a
:class:`HelperRequest`; running the op on meta tensors gives the result types, so
nothing is guessed from Helion's symbolic metadata. A ``?`` dim is a fresh size
symbol of a fake tensor instead (one ``FakeTensorMode`` for the process), and a
symbolic result dim is ``?``. The call names a helper
derived from the request. After every function is built,
:meth:`AtenHelperTable.materialize` lowers the requests not yet in the
process-wide cache in one torch-mlir run and clones the helpers into the module.
When that run fails, each request is lowered alone and the first failing node is
reported at its source line.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field
import hashlib
import operator
from typing import TYPE_CHECKING

from mlir.dialects import arith as arith_d
from mlir.dialects import func as func_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.shape_prop import TensorMetadata

from ..support import UnsupportedOperationError
from ..support import mlir_dtype_to_torch
from ..support import torch_dtype_to_mlir

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..build_context import BuildContext

ORIGINAL_ARGS = "helion_mlir_original_args"
"""Node meta key: ``(args, kwargs)`` before Helion's ``strip_unused_inputs`` (see inject)."""

_CACHE: dict[str, ir.Operation] = {}
# Keeps the parsed modules that own the cached helper functions alive.
_CACHE_MODULES: list[ir.Module] = []
_FAKE_MODE: FakeTensorMode | None = None


def _fake_mode() -> FakeTensorMode:
    """The fake tensor mode of samples with dynamic dims (its own shape env)."""
    global _FAKE_MODE
    if _FAKE_MODE is None:
        _FAKE_MODE = FakeTensorMode(shape_env=ShapeEnv())
    return _FAKE_MODE


def _mode_for(samples: list[object]) -> contextlib.AbstractContextManager:
    if any(isinstance(sample, FakeTensor) for sample in samples):
        return _fake_mode()
    return contextlib.nullcontext()


def is_aten_op(node: torch.fx.Node) -> bool:
    """An ATen ``OpOverload`` call with tensor result(s)."""
    if node.op != "call_function" or not isinstance(node.target, OpOverload):
        return False
    value = node.meta.get("val")
    if isinstance(value, (list, tuple)):
        return bool(value) and all(isinstance(item, torch.Tensor) for item in value)
    return isinstance(value, torch.Tensor)


def original_args(node: torch.fx.Node) -> tuple[tuple, dict]:
    """The node's arguments with the inputs Helion masked as ``None`` restored.

    Helion's ``strip_unused_inputs`` replaces repeated inputs (``x * x`` becomes
    ``mul(x, None)``) in place; :func:`install_original_args_capture` records the
    arguments just before, and only those ``None`` positions are filled back, so
    later rewrites of the node (e.g. inserted ``_mask_to``) are kept.
    """
    before_args, before_kwargs = node.meta.get(ORIGINAL_ARGS, ((), {}))
    args = tuple(
        _restore(arg, before_args[index] if index < len(before_args) else None)
        for index, arg in enumerate(node.args)
    )
    kwargs = {
        key: _restore(value, before_kwargs.get(key))
        for key, value in node.kwargs.items()
        if not key.startswith("_extra")
    }
    return args, kwargs


def _restore(current: object, before: object) -> object:
    if current is None and isinstance(before, torch.fx.Node):
        return before
    if (
        isinstance(current, (list, tuple))
        and isinstance(before, (list, tuple))
        and len(current) == len(before)
    ):
        return type(current)(map(_restore, current, before))
    return current


def install_original_args_capture() -> None:
    """Record node arguments before Helion's ``strip_unused_inputs`` masks them."""
    from helion._compiler.compile_environment import CompileEnvironment
    import helion._compiler.inductor_lowering as inductor_lowering

    from ..backend import MLIRBackend

    strip = inductor_lowering.strip_unused_inputs
    if getattr(strip, "_helion_mlir_capture", False):
        return

    def strip_unused_inputs(node: torch.fx.Node, *args: object) -> object:
        if isinstance(CompileEnvironment.current().backend, MLIRBackend):
            node.meta.setdefault(ORIGINAL_ARGS, (node.args, node.kwargs))
        return strip(node, *args)

    strip_unused_inputs._helion_mlir_capture = True
    inductor_lowering.strip_unused_inputs = strip_unused_inputs


@dataclass(frozen=True)
class _Operand:
    index: int

    def __repr__(self) -> str:
        return f"%{self.index}"


@dataclass(frozen=True)
class HelperRequest:
    target: OpOverload
    args: tuple
    kwargs: tuple[tuple[str, object], ...]
    operand_types: tuple[str, ...]
    result_types: tuple[str, ...]
    samples: tuple = field(compare=False, hash=False, repr=False)
    """Meta tensor or Python scalar standing in for each operand."""

    @property
    def name(self) -> str:
        key = repr((str(self.target), self.args, self.kwargs, self.operand_types))
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"_aten_{self.target.__name__.replace('.', '_')}_{digest}"

    @property
    def function_type(self) -> ir.FunctionType:
        return ir.FunctionType.get(
            [ir.Type.parse(text) for text in self.operand_types],
            [ir.Type.parse(text) for text in self.result_types],
        )


class AtenHelperTable:
    """The helper requests of one module."""

    def __init__(self) -> None:
        self._requests: dict[str, tuple[HelperRequest, torch.fx.Node]] = {}

    def add(self, request: HelperRequest, node: torch.fx.Node) -> None:
        self._requests.setdefault(request.name, (request, node))

    def materialize(self, module: ir.Module) -> None:
        """Define every requested helper in ``module`` (one torch-mlir run at most)."""
        missing = [
            (request, node)
            for request, node in self._requests.values()
            if request.name not in _CACHE
        ]
        if missing:
            _lower_into_cache(missing)
        with ir.InsertionPoint.at_block_begin(module.body):
            for request, node in self._requests.values():
                helper = _CACHE[request.name]
                actual = ir.TypeAttr(helper.attributes["function_type"]).value
                if str(actual) != str(request.function_type):
                    with _located(node):
                        raise UnsupportedOperationError(
                            str(request.target),
                            reason=(
                                f"torch-mlir lowered it to {actual}, but the call "
                                f"site needs {request.function_type}"
                            ),
                        )
                helper.clone().attributes["sym_visibility"] = ir.StringAttr.get(
                    "private"
                )


def lower_via_aten_helper(ctx: BuildContext, node: torch.fx.Node) -> object:
    """``func.call`` to the helper for this node's operand types."""
    return call_helper(ctx, node, node.target, *original_args(node))


def call_helper(
    ctx: BuildContext,
    node: torch.fx.Node,
    target: OpOverload,
    args: tuple,
    kwargs: dict,
) -> object:
    """``func.call`` to the helper for ``target(*args, **kwargs)``.

    Arguments may be FX nodes, MLIR values or literals; ``node`` locates errors.
    """
    args, kwargs, values, samples = _bind(ctx, node, args, kwargs, target)
    results = _evaluate(target, args, kwargs, samples)
    operands = [_as_operand(value) for value in values]
    request = HelperRequest(
        target,
        args,
        tuple(sorted(kwargs.items())),
        tuple(str(operand.type) for operand in operands),
        tuple(str(_tensor_type(result)) for result in results),
        tuple(samples),
    )
    ctx.aten_helpers.add(request, node)
    call = func_d.CallOp(
        [_tensor_type(result) for result in results], request.name, operands
    )
    return call.results[0] if len(call.results) == 1 else call


def infer_results(ctx: BuildContext, node: torch.fx.Node) -> tuple[torch.Tensor, ...]:
    """The node's results as meta tensors, computed from its operands' MLIR types."""
    args, kwargs, _, samples = _bind(ctx, node, *original_args(node), node.target)
    if node.op == "call_method":
        method = node.target

        def target(receiver: torch.Tensor, *rest: object, **options: object) -> object:
            return getattr(receiver, method)(*rest, **options)

    else:
        target = node.target
    return _evaluate(target, args, kwargs, samples)


def _evaluate(
    target: object, args: tuple, kwargs: dict, samples: list[object]
) -> tuple[torch.Tensor, ...]:
    args, kwargs = _substitute(args, samples), _substitute(kwargs, samples)
    try:
        with torch.no_grad(), _mode_for(samples):
            result = target(*args, **kwargs)
    except Exception as error:
        raise UnsupportedOperationError(
            str(target),
            reason=f"cannot infer its result for these operand types: {error}",
        ) from error
    return tuple(result) if isinstance(result, (list, tuple)) else (result,)


def _bind(
    ctx: BuildContext,
    node: torch.fx.Node,
    args: tuple,
    kwargs: dict,
    target: object = None,
) -> tuple[tuple, dict, list[ir.Value], list[object]]:
    """Replace inputs by literals or operand markers; collect operand values.

    A runtime scalar where ``target``'s schema takes a tensor is a 0-d tensor.
    """
    values: list[ir.Value] = []
    samples: list[object] = []

    def convert(arg: object) -> object:
        if isinstance(arg, (list, tuple)):
            return type(arg)(convert(item) for item in arg)
        if isinstance(arg, torch.fx.Node):
            value = ctx.get_value(arg)
            if value is None:
                literal = arg.meta.get("val")
                if isinstance(literal, (bool, int, float, torch.dtype)):
                    return literal
                raise UnsupportedOperationError(
                    str(node.target), reason=f"input {arg.name} has no lowered value"
                )
        elif isinstance(arg, ir.Value):
            value = arg
        else:
            return arg
        if not isinstance(value.type, ir.RankedTensorType):
            constant = _constant(value)
            if constant is not None:
                return constant
        values.append(value)
        samples.append(_sample(value.type))
        return _Operand(len(values) - 1)

    bound_args = convert(args)
    bound_kwargs = {key: convert(value) for key, value in kwargs.items()}
    schema = getattr(target, "_schema", None)
    for argument, arg in zip(
        schema.arguments if schema else [], bound_args, strict=False
    ):
        if (
            isinstance(arg, _Operand)
            and isinstance(argument.type, torch.TensorType)
            and not isinstance(values[arg.index].type, ir.RankedTensorType)
        ):
            scalar = _as_operand(values[arg.index])
            values[arg.index] = tensor_d.FromElementsOp(
                ir.RankedTensorType.get([], scalar.type), [scalar]
            ).result
            samples[arg.index] = _sample(values[arg.index].type)
    if any(isinstance(sample, FakeTensor) for sample in samples):
        # One mode for every tensor operand of the op.
        samples = [
            _sample(value.type, fake=True)
            if isinstance(value.type, ir.RankedTensorType)
            else sample
            for value, sample in zip(values, samples, strict=True)
        ]
    return bound_args, bound_kwargs, values, samples


def _substitute(structure: object, samples: list[object]) -> object:
    if isinstance(structure, _Operand):
        return samples[structure.index]
    if isinstance(structure, dict):
        return {key: _substitute(value, samples) for key, value in structure.items()}
    if isinstance(structure, (list, tuple)):
        return type(structure)(_substitute(item, samples) for item in structure)
    return structure


def _sample(value_type: ir.Type, *, fake: bool = False) -> object:
    """A meta tensor or Python scalar standing in for an operand of ``value_type``;
    a fake tensor with a fresh size symbol per ``?`` dim if it has any (or ``fake``)."""
    if isinstance(value_type, ir.RankedTensorType):
        element = value_type.element_type
        dtype = (
            torch.int64
            if isinstance(element, ir.IndexType)
            else mlir_dtype_to_torch(str(element), default=None)
        )
        if dtype is None:
            raise UnsupportedOperationError(
                "ATen helper operand", reason=f"unsupported element type {element}"
            )
        shape = list(value_type.shape)
        if not fake and not any(ir.ShapedType.is_dynamic_size(dim) for dim in shape):
            return torch.empty(shape, dtype=dtype, device="meta")
        mode = _fake_mode()
        sizes = []
        for dim in shape:
            if ir.ShapedType.is_dynamic_size(dim):
                dim = mode.shape_env.create_unbacked_symint()
                torch._check(dim >= 0)
            sizes.append(dim)
        with mode:
            return torch.empty(sizes, dtype=dtype)
    if isinstance(value_type, ir.IntegerType) and value_type.width == 1:
        return True
    if isinstance(value_type, (ir.IndexType, ir.IntegerType)):
        return 1
    if isinstance(value_type, ir.FloatType):
        return 1.0
    raise UnsupportedOperationError(
        "ATen helper operand", reason=f"unsupported scalar type {value_type}"
    )


def _as_operand(value: ir.Value) -> ir.Value:
    """Index tensors as ``i64``; runtime scalars as ``i1``, ``i64`` or ``f64``."""
    value_type = value.type
    i64 = ir.IntegerType.get_signless(64)
    if isinstance(value_type, ir.RankedTensorType):
        if isinstance(value_type.element_type, ir.IndexType):
            return arith_d.IndexCastOp(
                ir.RankedTensorType.get(list(value_type.shape), i64), value
            ).result
        return value
    if isinstance(value_type, ir.IndexType):
        return arith_d.IndexCastOp(i64, value).result
    if isinstance(value_type, ir.IntegerType):
        if 1 < value_type.width < 64:
            return arith_d.ExtSIOp(i64, value).result
        return value
    f64 = ir.F64Type.get()
    return value if value_type == f64 else arith_d.ExtFOp(f64, value).result


def _constant(value: ir.Value) -> bool | int | float | None:
    owner = value.owner
    if not isinstance(owner, ir.OpView) or owner.name != "arith.constant":
        return None
    attribute = owner.attributes["value"]
    if isinstance(value.type, ir.FloatType):
        return ir.FloatAttr(attribute).value
    integer = ir.IntegerAttr(attribute).value
    if isinstance(value.type, ir.IntegerType) and value.type.width == 1:
        return bool(integer)
    return integer


def _tensor_type(result: object) -> ir.RankedTensorType:
    if not isinstance(result, torch.Tensor):
        raise UnsupportedOperationError(
            "ATen helper result", reason=f"non-tensor result {type(result).__name__}"
        )
    return ir.RankedTensorType.get(
        [static_dim(dim) for dim in result.shape], torch_dtype_to_mlir(result.dtype)
    )


def static_dim(size: int | torch.SymInt) -> int:
    """A result dim as an MLIR dim: the dynamic size sentinel if it is symbolic."""
    if isinstance(size, torch.SymInt):
        expr = size.node.expr
        return ir.ShapedType.get_dynamic_size() if expr.free_symbols else int(expr)
    return int(size)


@contextlib.contextmanager
def _located(node: torch.fx.Node) -> Iterator[None]:
    location = node.meta.get("location")
    with location if location is not None else contextlib.nullcontext():
        yield


def _lower_into_cache(requests: list[tuple[HelperRequest, torch.fx.Node]]) -> None:
    try:
        _cache(_run_torch_mlir([request for request, _ in requests]))
        return
    except Exception:
        pass
    for request, node in requests:
        try:
            _cache(_run_torch_mlir([request]))
        except Exception as error:
            message = str(error).strip().splitlines()
            with _located(node):
                raise UnsupportedOperationError(
                    str(request.target),
                    reason=(
                        f"torch-mlir cannot lower it for operand types "
                        f"{list(request.operand_types)}: "
                        f"{message[0] if message else type(error).__name__}"
                    ),
                ) from error


def _cache(helpers: ir.Module) -> None:
    _CACHE_MODULES.append(helpers)
    for operation in helpers.body.operations:
        if "sym_name" in operation.attributes:
            _CACHE[ir.StringAttr(operation.attributes["sym_name"]).value] = operation


def _run_torch_mlir(requests: list[HelperRequest]) -> ir.Module:
    """Import and lower ``requests`` with torch-mlir; the result lives in this context."""
    from torch_mlir.compiler_utils import OutputType
    from torch_mlir.compiler_utils import lower_mlir_module
    from torch_mlir.compiler_utils import run_pipeline_with_repro_report
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter
    import torch_mlir.ir as tm_ir

    context = tm_ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    for request in requests:
        importer.import_stateless_graph(_fx_graph(request), func_name=request.name)
    run_pipeline_with_repro_report(
        importer.module,
        "builtin.module(func.func(torch-match-quantized-custom-ops),"
        " torchdynamo-export-to-torch-backend-pipeline{})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )
    lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, importer.module)
    return ir.Module.parse(
        importer.module.operation.get_asm(binary=False, enable_debug_info=False)
    )


def _fx_graph(request: HelperRequest) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    placeholders = []
    for index, sample in enumerate(request.samples):
        placeholder = graph.placeholder(f"arg{index}")
        _set_meta(placeholder, sample)
        placeholders.append(placeholder)
    args = _substitute(request.args, placeholders)
    kwargs = _substitute(dict(request.kwargs), placeholders)
    call = graph.call_function(request.target, args, kwargs)
    with _mode_for(list(request.samples)):
        results = request.target(
            *_substitute(request.args, list(request.samples)),
            **_substitute(dict(request.kwargs), list(request.samples)),
        )
    if isinstance(results, torch.Tensor):
        _set_meta(call, results)
        graph.output((call,))
        return graph
    call.meta["val"] = tuple(results)
    outputs = []
    for index, result in enumerate(results):
        item = graph.call_function(operator.getitem, (call, index))
        _set_meta(item, result)
        outputs.append(item)
    graph.output(tuple(outputs))
    return graph


def _set_meta(node: torch.fx.Node, value: object) -> None:
    node.meta["val"] = value
    if isinstance(value, torch.Tensor):
        node.meta["tensor_meta"] = TensorMetadata(
            value.shape, value.dtype, False, value.stride(), None, False, {}
        )
