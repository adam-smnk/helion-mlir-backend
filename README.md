# helion-mlir

External MLIR backend package for vanilla Helion.

This package keeps Helion unmodified and adds MLIR support out-of-tree.

When `helion_mlir_backend` is imported, it registers the `mlir` backend in Helion via Helion's native backend registry.

Use MLIR APIs directly from this package:

- `from helion_mlir_backend import generate_mlir`
- `settings.backend = "mlir"` in Helion flows

## Install

```bash
uv sync
```

Or with pip:

```bash
pip install .
```

## Enable MLIR Backend

Any import from `helion_mlir_backend` enables registration.

Recommended:

```python
from helion_mlir_backend import generate_mlir
```

Equivalent explicit form:

```python
import helion_mlir_backend  # side effect: register backend="mlir"
from helion_mlir_backend import generate_mlir
```

This works whether `helion` was imported before or after `helion_mlir_backend`.

## Quick Check

```bash
python - <<'PY'
from helion_mlir_backend import generate_mlir
from helion._compiler.backend_registry import get_backend_class
print("generate_mlir import OK:", callable(generate_mlir))
print("mlir backend module:", get_backend_class("mlir").__module__)
PY
```

Expected backend module prefix:

- `helion_mlir_backend._compiler.mlir.backend`

## Notes

- No Helion source patching is required.
- No system-wide or user-global Python startup files are modified.
- No runtime `helion.mlir` module shim is used.
