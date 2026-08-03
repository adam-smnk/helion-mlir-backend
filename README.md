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

## Troubleshooting

### `ModuleNotFoundError: No module named "mlir"`

The MLIR Python bindings are missing from your environment.

- If using this repo workflow, run:

```bash
uv sync
```

- If using pip-based setup, ensure `mlir-python-bindings` is installed in the
	same environment as `helion` and `helion-mlir`.

### `torch-mlir`/`torch` compatibility errors

`torch-mlir` wheels are version-sensitive with respect to `torch`.

- Use the pinned versions from `pyproject.toml` in this repository.
- Recreate or resync your environment if versions drift:

```bash
uv sync
```

- Quick check:

```bash
python - <<'PY'
import torch
import torch_mlir
print('torch:', torch.__version__)
print('torch_mlir:', getattr(torch_mlir, '__version__', 'unknown'))
PY
```

## Debug Environment Variables

Set any of these to a truthy value (`1`, `true`, `yes`) to dump MLIR IR to
stdout at the corresponding pipeline stage.

| Variable | Dump point |
|---|---|
| `HELION_MLIR_DUMP_IR` | Raw generated IR before inlining (shows `@_aten_*` helpers and tile loops) |
| `HELION_MLIR_DUMP_PRE_LOWERING` | After inlining / all pre-passes, immediately before the lighthouse lowering schedule |
| `HELION_MLIR_DUMP_LOWERED` | After lighthouse lowering (LLVM dialect, expanded memref descriptors) |

## Development

This repo uses pre-commit + Ruff rules aligned with Helion style.

Install hooks:

```bash
uv run pre-commit install
```

Run hooks on all files:

```bash
uv run pre-commit run -a
```

Run Ruff directly:

```bash
uv run ruff check helion_mlir_backend tests
uv run ruff format helion_mlir_backend tests
```

Run MLIR test suites:

```bash
uv run --with pytest pytest tests/test_mlir_backend.py tests/test_mlir_integration.py -q
```
