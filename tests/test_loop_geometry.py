"""Loop geometry from Helion metadata: begin/end/step, tile positions, forall shape."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import opt_pipeline

from helion_mlir_backend import generate_mlir


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4))
def root_offset_1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(4, x.size(0)):
        out[tile] = x[tile] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 8))
def root_offset_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm, tn in hl.tile([4, 8], x.size()):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def nested_offset(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(2, n):
            out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def nested_offset_end(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(4, n - 4):
            out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4))
def nested_grid_step(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm in hl.tile(x.size(0)):
        for j in hl.grid(0, x.size(1), 2):
            out[tm, j] = x[tm, j] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4))
def root_grid_step(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for i in hl.grid(1, x.size(0), 3):
        out[i, :] = x[i, :] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_begin_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(8, x.size(0)):
        out[tile] = x[tile] + tile.begin
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_end_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(8, x.size(0)):
        out[tile] = x[tile] + tile.end
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_id_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(8, x.size(0)):
        out[tile] = x[tile] + tile.id
    return out


def _add_one(x: torch.Tensor) -> torch.Tensor:
    return x + 1.0


def _per_tile(value):
    """Reference for ``hl.tile(8, n)`` adding ``value(begin, end)`` to each tile."""

    def reference(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        for begin in range(8, x.size(0), 8):
            end = min(begin + 8, x.size(0))
            out[begin:end] += value(begin, end)
        return out

    return reference


# Outputs start as zeros_like, which the backend does not yet preserve (I13), so only
# the region each kernel writes is compared.
@pytest.mark.parametrize(
    ("kernel", "reference", "shape", "region"),
    [
        pytest.param(root_offset_1d, _add_one, (20,), (slice(4, None),), id="root_1d"),
        pytest.param(
            root_offset_2d,
            _add_one,
            (12, 24),
            (slice(4, None), slice(8, None)),
            id="root_2d",
        ),
        pytest.param(
            nested_offset,
            _add_one,
            (16, 10),
            (slice(None), slice(2, None)),
            id="nested_begin",
        ),
        pytest.param(
            nested_offset_end,
            _add_one,
            (16, 16),
            (slice(None), slice(4, 12)),
            id="nested_begin_end",
        ),
        pytest.param(
            nested_grid_step,
            _add_one,
            (8, 6),
            (slice(None), slice(None, None, 2)),
            id="nested_grid_step",
        ),
        pytest.param(
            root_grid_step,
            _add_one,
            (10, 4),
            (slice(1, None, 3),),
            id="root_grid_step",
        ),
        pytest.param(
            tile_begin_kernel,
            _per_tile(lambda begin, end: begin),
            (32,),
            (slice(8, None),),
            id="tile_begin",
        ),
        pytest.param(
            tile_end_kernel,
            _per_tile(lambda begin, end: end),
            (32,),
            (slice(8, None),),
            id="tile_end",
        ),
        pytest.param(
            tile_id_kernel,
            _per_tile(lambda begin, end: begin // 8),
            (32,),
            (slice(8, None),),
            id="tile_id",
        ),
    ],
)
def test_loop_geometry(kernel, reference, shape, region) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        reference,
        [torch.randn(*shape)],
        paths=("direct", "generated"),
        region=region,
    )


def test_outer_forall_is_normalized() -> None:
    module = generate_mlir(root_offset_2d, [torch.randn(12, 24)])
    text = str(module)
    assert "scf.forall (%arg1, %arg2) in (2, 2)" in text
    assert "affine_map<(d0) -> (d0 * 4 + 4)>" in text
    assert "affine_map<(d0) -> (d0 * 8 + 8)>" in text


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(32, 32, 32))
def f32_matmul_32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@pytest.mark.isolated
@pytest.mark.slow
def test_sfc_remap_applies_to_normalized_forall() -> None:
    """The optimizing pipeline's SFC remap only rewrites a normalized 2-D forall."""
    from lighthouse.pipeline.driver import BackendDriver
    import mlir.ir as ir

    from helion_mlir_backend._compiler.execution import inline_module
    from helion_mlir_backend._compiler.execution import pipeline_descriptor

    module = inline_module(
        generate_mlir(f32_matmul_32, [torch.randn(128, 64), torch.randn(64, 128)])
    )
    remapped = False
    with module.context, ir.Location.unknown():
        driver = BackendDriver(
            module, "f32_matmul_32", result_to_args=True, benchmark=False
        )
        driver.add_stage(pipeline_descriptor(optimized=True))
        for stage in driver.stages:
            before = str(module)
            module = stage.apply(module)
            if "sfc_remap_forall" in str(stage):
                # A 4x4 tile grid is remapped through a 16-entry index table.
                remapped = str(module) != before and "tensor<16xi64>" in str(module)
                break
    assert remapped


@pytest.mark.isolated
@pytest.mark.slow
def test_sfc_matmul_executes_under_opt_pipeline() -> None:
    x, y = torch.randn(128, 64), torch.randn(64, 128)
    with opt_pipeline():
        result = f32_matmul_32(x, y)
    torch.testing.assert_close(result, x @ y, atol=1e-4, rtol=1e-4)
