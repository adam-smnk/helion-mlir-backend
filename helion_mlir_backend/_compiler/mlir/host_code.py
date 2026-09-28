"""The kernel's host code as a re-executable generator function.

Helion's own codegen keeps the kernel's host statements as real Python that
runs on every call around the device kernel launch (see
``helion._compiler.generate_ast``). The MLIR backend does the same: the host
function is a copy of the kernel's AST body in which the top-level device loops
(``ast.For`` tagged ``LoopType.GRID``; Helion requires them at top level, with
only ``hl.barrier()`` between them) become one ``yield locals()``, and every
host-side block size (``hl.register_block_size``) becomes the config's value,
as in Helion's host codegen. The driver runs it to the ``yield``, calls the
compiled kernel with tensors taken from the yielded locals, then resumes it; the
generator's return value is the kernel's return value.
"""

from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion._compiler.host_function import HostFunction


def build_host_function(
    hf: HostFunction, block_size: Callable[[int], int]
) -> Callable[..., Generator[dict[str, object], None, object]]:
    """Compile *hf*'s host code; ``hf.body`` is not modified."""
    body: list[ast.stmt] = []
    yielded = False
    for stmt in hf.body:
        if _is_device_statement(stmt):
            if not yielded:
                body.append(ast.copy_location(_yield_locals(), stmt))
                yielded = True
            continue
        body.append(_HostCode(block_size).visit(stmt))
    if not yielded:
        body.append(_yield_locals())

    func_name = f"_helion_mlir_host_{hf.name}"
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

    code = compile(module, filename=f"<helion-mlir-host:{hf.name}>", mode="exec")
    namespace: dict[str, object] = dict(hf.fn.__globals__)
    # Host expressions of globals go through these aliases (``_source_module.W``).
    namespace.update(
        {alias: imported.value for alias, imported in hf.global_imports.items()}
    )
    exec(code, namespace)
    return namespace[func_name]  # type: ignore[return-value]


def _is_device_statement(stmt: ast.stmt) -> bool:
    from helion._compiler.ast_extension import LoopType
    from helion._compiler.type_info import BarrierResultType

    if isinstance(stmt, ast.For):
        return getattr(stmt, "_loop_type", None) == LoopType.GRID
    return isinstance(stmt, ast.Expr) and isinstance(
        getattr(stmt.value, "_type_info", None), BarrierResultType
    )


def _yield_locals() -> ast.stmt:
    call = ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])
    return ast.Expr(value=ast.Yield(value=call))


class _HostCode(ast.NodeTransformer):
    """Copy a host statement, replacing block-size expressions by config values.

    Every visited node is shallow-copied (with its list fields) before its
    children are transformed, so Helion's own AST stays intact.
    """

    def __init__(self, block_size: Callable[[int], int]) -> None:
        self.block_size = block_size

    def generic_visit(self, node: ast.AST) -> ast.AST:
        from helion._compiler.ast_extension import ExtendedAST

        node = node.copy() if isinstance(node, ExtendedAST) else copy.copy(node)
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                setattr(node, field, list(value))
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        from helion._compiler.type_info import BlockSizeType
        from helion._compiler.type_info import SequenceType
        from helion._compiler.type_info import TileIndexType

        block_types = (BlockSizeType, TileIndexType)
        type_info = getattr(node, "_type_info", None)
        if isinstance(type_info, block_types):
            return ast.copy_location(
                ast.Constant(self.block_size(type_info.block_id)), node
            )
        if isinstance(type_info, SequenceType) and all(
            isinstance(item, block_types) for item in type_info.unpack()
        ):
            sizes = [
                ast.Constant(self.block_size(item.block_id))
                for item in type_info.unpack()
            ]
            return ast.copy_location(ast.List(elts=sizes, ctx=ast.Load()), node)
        return self.generic_visit(node)
