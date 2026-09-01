"""Typed table for generic torch-mlir ATen helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir


class AtenHelperTable:
    """Own helper-function metadata used by generic ATen lowering."""

    def __init__(self, module: ir.Module | None = None) -> None:
        self.module = module
        self.entries: dict[int, tuple[str, list[ir.Type]]] = {}

    def replace(self, entries: dict[int, tuple[str, list[ir.Type]]]) -> None:
        """Replace the node-to-helper mapping produced by the pre-pass."""
        self.entries = entries

    def get(self, node_id: int) -> tuple[str, list[ir.Type]] | None:
        """Return helper metadata for an FX node identity."""
        return self.entries.get(node_id)

    def signature_matches(self, func_name: str, values: list[ir.Value]) -> bool:
        """Return whether a helper's input types match call-site values."""
        import mlir.ir as ir

        if self.module is None:
            return False
        for operation in self.module.body.operations:
            name_attr = operation.attributes.get("sym_name")
            if name_attr is None:
                continue
            name = name_attr.value if hasattr(name_attr, "value") else str(name_attr)
            if name != func_name:
                continue
            function_type = ir.FunctionType(
                ir.TypeAttr(operation.attributes["function_type"]).value
            )
            return len(function_type.inputs) == len(values) and all(
                str(expected) == str(actual.type)
                for expected, actual in zip(function_type.inputs, values, strict=True)
            )
        return False
