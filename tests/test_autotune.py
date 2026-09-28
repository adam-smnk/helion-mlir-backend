"""Autotuning, the pipeline config key and the JIT cache (plan Phase 10)."""

from __future__ import annotations

import random
from unittest import mock

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend import compile_mlir
from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.mlir.autotune import cpu_name
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.backend import raise_block_minimums
from helion_mlir_backend.api import _compile


def axpy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


def add_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@pytest.fixture
def tuning(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Allow autotuning, with a fresh cache directory."""
    monkeypatch.setenv("HELION_DISALLOW_AUTOTUNING", "0")
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HELION_MLIR_PIPELINE", "0")


def test_one_config_is_used_as_is(tuning: None) -> None:
    config = helion.Config(block_sizes=[16, 32])
    kernel = helion.kernel(backend="mlir", config=config)(axpy)
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    with mock.patch.object(MLIRBackend, "autotune") as autotune:
        check_kernel(kernel, lambda x, y: x * 2.0 + y, [x, y])
    autotune.assert_not_called()
    (bound,) = kernel._bound_kernels.values()
    assert bound._config == config
    assert len(bound._compile_cache) == 1


def test_several_configs_are_searched(tuning: None) -> None:
    configs = [helion.Config(block_sizes=[16, 16]), helion.Config(block_sizes=[32, 64])]
    kernel = helion.kernel(backend="mlir", configs=configs)(axpy)
    check_kernel(
        kernel, lambda x, y: x * 2.0 + y, [torch.randn(64, 64), torch.randn(64, 64)]
    )
    (bound,) = kernel._bound_kernels.values()
    assert bound._config in configs
    assert set(configs) <= set(bound._compile_cache)


@pytest.mark.slow
def test_full_search_and_cpu_autotune_cache(tuning: None, tmp_path: object) -> None:
    x, y = torch.randn(256), torch.randn(256)
    kernel = helion.kernel(backend="mlir", autotune_effort="quick")(add_1d)
    check_kernel(kernel, torch.add, [x, y])
    (bound,) = kernel._bound_kernels.values()
    assert bound._config is not None
    assert list(tmp_path.glob("*.best_config"))

    cached = helion.kernel(backend="mlir", autotune_effort="quick")(add_1d)
    with mock.patch("helion.autotuner.LFBOTreeSearch.autotune") as search:
        check_kernel(cached, torch.add, [x, y])
    search.assert_not_called()
    (bound_again,) = cached._bound_kernels.values()
    assert bound_again._config == bound._config


def test_autotuning_is_disallowed_in_tests() -> None:
    kernel = helion.kernel(backend="mlir")(add_1d)
    with pytest.raises(helion.exc.AutotuningDisallowedInEnvironment):
        kernel(torch.randn(64), torch.randn(64))


def test_cpu_name_is_known() -> None:
    assert cpu_name()


def test_opt_search_raises_small_block_sizes() -> None:
    x, y = torch.randn(256, 256), torch.randn(256, 256)
    _, _, env = _compile(helion.kernel(backend="mlir")(matmul), [x, y], None)
    spec = env.config_spec
    assert min(spec.default_config().config["block_sizes"]) < 32
    raise_block_minimums(spec)
    assert min(spec.default_config().config["block_sizes"]) >= 32


def test_given_configs_keep_their_block_sizes_on_the_opt_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_MLIR_PIPELINE", "1")
    config = helion.Config(block_sizes=[1, 16, 32])
    kernel = helion.kernel(backend="mlir", config=config)(matmul)
    text = str(generate_mlir(kernel, [torch.randn(64, 64), torch.randn(64, 64)]))
    assert "tensor<1x16xf32>" in text


def test_block_size_prior_prefers_divisors() -> None:
    x, y = torch.randn(96, 96), torch.randn(96, 96)
    _, _, env = _compile(helion.kernel(backend="mlir")(axpy), [x, y], None)
    prior = MLIRBackend().config_value_priors(env.config_spec)["block_sizes"]
    fragment = mock.Mock(low=16, high=128)
    random.seed(0)
    samples = [prior(fragment, 0) for _ in range(1000)]
    assert set(samples) <= {16, 32, 64, 128}
    assert sum(96 % size == 0 for size in samples) > 700


def test_config_selects_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_MLIR_PIPELINE", "0")
    chosen: list[str] = []
    original = execution.pipeline_descriptor

    def record(pipeline: str | None = None) -> object:
        chosen.append(pipeline)
        return original("scalar")

    monkeypatch.setattr(execution, "pipeline_descriptor", record)
    monkeypatch.setattr(execution, "_JIT_CACHE", type(execution._JIT_CACHE)())
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    config = helion.Config(block_sizes=[32, 32], mlir_pipeline="opt")
    run = compile_mlir(helion.kernel(backend="mlir")(axpy), [x, y], config=config)
    torch.testing.assert_close(run(x, y), x * 2.0 + y)
    assert chosen == ["opt"]


def test_unknown_pipeline_in_config_is_rejected() -> None:
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    config = helion.Config(block_sizes=[32, 32], mlir_pipeline="fast")
    with pytest.raises(ValueError, match="unknown pipeline 'fast'"):
        compile_mlir(helion.kernel(backend="mlir")(axpy), [x, y], config=config)


def test_identical_modules_are_compiled_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "_JIT_CACHE", type(execution._JIT_CACHE)())
    compiled = []
    original = execution._compile_entry

    def count(*args: object) -> object:
        compiled.append(args[1])
        return original(*args)

    monkeypatch.setattr(execution, "_compile_entry", count)
    x, y = torch.randn(40, 40), torch.randn(40, 40)
    config = helion.Config(block_sizes=[32, 32])
    for kernel in (
        helion.kernel(backend="mlir")(axpy),
        helion.kernel(backend="mlir")(axpy),
    ):
        run = compile_mlir(kernel, [x, y], config=config)
        torch.testing.assert_close(run(x, y), x * 2.0 + y)
    assert compiled == ["axpy"]
