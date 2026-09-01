"""Build a real, re-executable "host prefix" function from a compiled kernel.

Helion's own Triton codegen path keeps the kernel's "outside the loop" host
statements as real Python source (see ``helion._compiler.generate_ast``),
re-executed with real tensors on every call before launching the compiled
device kernel. The MLIR backend bypasses that machinery entirely (see
``codegen.py``'s module docstring), so any host-side tensor computation that
isn't one of the kernel's own declared parameters (or a view/reshape of one)
has no way to reach the compiled MLIR function -- see
``lowering/host_tensor_ops.py::lower_host_tensor``.

This module builds a narrow, MLIR-backend-only substitute: a standalone
function derived from the kernel's own AST body, with every
``hl.tile``/``hl.grid`` loop (``ast.For`` nodes tagged ``LoopType.GRID`` by
Helion's own type-propagation pass) replaced by a no-op placeholder, since
those are compiled to MLIR separately. Calling the result with the kernel's
real positional arguments re-executes every other host statement for real --
exactly like Helion's own Triton wrapper does -- and returns every local
variable bound during that call, so a caller can pick out whichever host
tensors (or the kernel's own return value) it needs.

This is intentionally a narrow, self-contained primitive: it does not decide
*which* names a caller needs, and it does not itself drive MLIR compilation
or execution -- see ``bound_kernel.py`` for how it is wired into the actual
kernel-call path.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from helion._compiler.host_function import HostFunction

_LOCALS_SENTINEL = "_helion_mlir_host_prefix_return_value"


class UnsupportedHostPrefixError(Exception):
    """Raised when a kernel's host-level control flow isn't supported yet."""


def build_host_prefix_function(hf: HostFunction) -> Callable[..., dict[str, object]]:
    """Compile a standalone function that runs *hf*'s real host statements.

    The returned callable takes the kernel's own positional/keyword arguments
    (same signature as the original kernel function) and returns a
    ``dict[str, object]`` of every local variable bound during that call
    (including, under ``_LOCALS_SENTINEL``, the value the kernel's own
    ``return`` statement would have produced, if any).

    Mutates ``hf.body`` in place while transforming it (no deep copy -- the
    ``ExtendedAST`` mixin Helion's own AST nodes use isn't deepcopy-safe).
    ``hf`` is expected to be a fresh, single-use compile artifact whose
    ``device_ir`` (already built from the original AST) is unaffected by
    mutating ``body`` afterward.
    """
    body = [_NeutralizeGridLoops().visit(stmt) for stmt in hf.body]
    body = _capture_return_value(body)
    for stmt in body:
        ast.fix_missing_locations(stmt)

    func_name = f"_helion_mlir_host_prefix_{hf.name}"
    func_def = ast.FunctionDef(
        name=func_name,
        args=hf.definition.args,
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        lineno=1,
        col_offset=0,
    )
    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)

    code = compile(module, filename=f"<helion-mlir-host-prefix:{hf.name}>", mode="exec")
    namespace: dict[str, object] = dict(hf.fn.__globals__)
    exec(code, namespace)
    return namespace[func_name]  # type: ignore[return-value]


class _NeutralizeGridLoops(ast.NodeTransformer):
    """Replace every ``hl.tile``/``hl.grid`` for-loop with a no-op placeholder.

    Recurses through the whole body (not just top-level statements) so a
    GRID loop nested inside a host-level ``if``/``while``/``try`` is still
    found. Every other statement -- including ordinary host ``for``/``while``
    loops -- is left untouched and re-executed for real. ``hl.barrier()``
    calls are also neutralized: they're a real Python function that only
    raises when actually called (it's meant to be traced, not executed), and
    the phase-sequencing this AST feeds into (see ``bound_kernel.py``'s
    multi-phase driver) already provides the ordering a barrier would.
    """

    def visit_For(self, node: ast.For) -> ast.AST:
        from helion._compiler.ast_extension import LoopType

        if getattr(node, "_loop_type", None) == LoopType.GRID:
            placeholder = ast.Pass()
            return ast.copy_location(placeholder, node)
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        from helion._compiler.type_info import BarrierResultType

        if isinstance(getattr(node.value, "_type_info", None), BarrierResultType):
            placeholder = ast.Pass()
            return ast.copy_location(placeholder, node)
        self.generic_visit(node)
        return node


def _capture_return_value(body: list[ast.stmt]) -> list[ast.stmt]:
    """Rewrite the kernel's own ``return`` (if any) so its value survives.

    Only a single ``return`` as the body's very last statement is supported
    in V1: rewriting an early/conditional return into a plain assignment
    would change which statements execute afterward. Appends a final
    ``return locals()`` so every local variable -- including the captured
    return value under ``_LOCALS_SENTINEL`` -- comes back to the caller.
    """
    returns = [
        stmt
        for stmt in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(stmt, ast.Return)
    ]
    if len(returns) > 1 or (returns and returns[0] is not body[-1]):
        raise UnsupportedHostPrefixError(
            "host-prefix driver only supports a kernel whose sole `return` "
            "statement (if any) is its very last top-level statement"
        )

    new_body = list(body)
    if returns:
        (return_stmt,) = returns
        assert isinstance(return_stmt, ast.Return)
        value = (
            return_stmt.value
            if return_stmt.value is not None
            else ast.Constant(value=None)
        )
        assign = ast.Assign(
            targets=[ast.Name(id=_LOCALS_SENTINEL, ctx=ast.Store())],
            value=value,
        )
        ast.copy_location(assign, return_stmt)
        new_body[-1] = assign

    locals_call = ast.Call(
        func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]
    )
    new_body.append(ast.Return(value=locals_call))
    return new_body
