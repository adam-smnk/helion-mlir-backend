"""Pytest configuration for the MLIR backend suite."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

import pytest

_CHILD_ENV = "HELION_MLIR_ISOLATED_CHILD"

_CHILD_SCRIPT = """
import importlib, pickle, sys
sys.path.insert(0, sys.argv[1])
module = importlib.import_module(sys.argv[2])
with open(sys.argv[4], "rb") as handle:
    kwargs = pickle.load(handle)
getattr(module, sys.argv[3])(**kwargs)
"""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="re-record tests/golden/*.txt IR signatures instead of comparing",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "isolated: run the test body in a fresh subprocess so a native crash only "
        "fails that test",
    )
    config.addinivalue_line("markers", "slow: long-running test")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if pyfuncitem.get_closest_marker("isolated") is None or os.environ.get(_CHILD_ENV):
        return None
    if pyfuncitem.cls is not None:
        raise pytest.UsageError(
            "@pytest.mark.isolated supports module-level tests only"
        )
    kwargs = {
        name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
        pickle.dump(kwargs, handle)
        kwargs_path = handle.name
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD_SCRIPT,
                str(pyfuncitem.config.rootpath),
                pyfuncitem.module.__name__,
                pyfuncitem.originalname,
                kwargs_path,
            ],
            env={**os.environ, _CHILD_ENV: "1"},
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(kwargs_path)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-25:])
        how = (
            f"crashed with signal {-completed.returncode}"
            if completed.returncode < 0
            else f"failed (exit code {completed.returncode})"
        )
        pytest.fail(f"isolated test {how}:\n{tail}", pytrace=False)
    return True
