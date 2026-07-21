"""Registers MLIR backend per-op codegen modules.

Importing this file triggers any ``@_decorators.codegen(op, "mlir")``
registrations for the MLIR backend.  Currently the MLIR backend overrides
the entire codegen step (via :class:`~helion._compiler.mlir.codegen.MLIRModuleBuilder`)
rather than individual per-op hooks, so this file is intentionally minimal.
"""
# Future: add per-op MLIR codegen registrations here.
