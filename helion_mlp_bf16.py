"""BF16 MLP benchmark for Helion's MLIR backend.

The Helion path uses the mmt4d-style AMX matmul shape from
``helion_matmul_bf16.py`` and fuses Linear bias/ReLU into the final bf16 store.
Only end-to-end model latency is reported.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import helion_block_pack
import torch
from torch import Tensor
import torch.nn as nn

import helion_mlir_backend  # noqa: F401

logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)

if TYPE_CHECKING:
    from collections.abc import Callable


BATCH_SIZE = int(os.environ.get("HELION_MLP_BATCH", "4096"))
FEATURE_SIZE = int(os.environ.get("HELION_MLP_FEATURES", "4096"))
HIDDEN_SIZE = int(os.environ.get("HELION_MLP_HIDDEN", str(FEATURE_SIZE)))
OUTPUT_SIZE = int(os.environ.get("HELION_MLP_OUTPUT", str(FEATURE_SIZE)))
BLOCK_M = int(os.environ.get("HELION_MLP_BLOCK_M", "32"))
BLOCK_N = int(os.environ.get("HELION_MLP_BLOCK_N", "32"))
BLOCK_K = int(os.environ.get("HELION_MLP_BLOCK_K", "32"))
WARMUP_ITERS = int(os.environ.get("HELION_MLP_WARMUP", "3"))
BENCHMARK_ITERS = int(os.environ.get("HELION_MLP_ITERS", "8"))
SAMPLES = int(os.environ.get("HELION_MLP_SAMPLES", "5"))


class Model(nn.Module):
    def __init__(
        self, input_size: int, layer_sizes: list[int], output_size: int
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.extend([nn.Linear(current_input_size, layer_size), nn.ReLU()])
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
)
def pack_a_mmt4d_kernel(a4_src: Tensor) -> Tensor:
    blocks_m, block_m, blocks_k, block_k = a4_src.shape
    out = torch.empty(
        (blocks_m, blocks_k, block_m, block_k),
        dtype=a4_src.dtype,
        device=a4_src.device,
    )
    for block_mi, block_ki, tile_m, tile_k in hl.tile(
        [blocks_m, blocks_k, block_m, block_k]
    ):
        out[block_mi, block_ki, tile_m, tile_k] = a4_src[
            block_mi, tile_m, block_ki, tile_k
        ].permute(0, 2, 1, 3)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def linear_bf16_mmt4d_mlir(
    a4: Tensor,
    b4: Tensor,
    bias2: Tensor,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"

    out = torch.empty(
        (blocks_m, block_m, blocks_n, block_n),
        dtype=a4.dtype,
        device=a4.device,
    )
    for tile_blocks_m, tile_blocks_n in hl.tile([blocks_m, blocks_n]):
        acc = hl.zeros(
            [tile_blocks_m, tile_blocks_n, block_m, block_n], dtype=torch.float32
        )
        acc = acc + torch.einsum(
            "akmc,bkcn->abmn",
            a4[tile_blocks_m, :, :, :],
            b4[tile_blocks_n, :, :, :],
        )
        y = acc.permute(0, 2, 1, 3) + bias2[tile_blocks_n, :]
        out[tile_blocks_m, :, tile_blocks_n, :] = epilogue(y).to(a4.dtype)
    return out


def identity_epilogue(x: Tensor) -> Tensor:
    return x


def relu_epilogue(x: Tensor) -> Tensor:
    return torch.relu(x)


def _check_divisible(name: str, value: int, block: int) -> None:
    if value % block:
        raise ValueError(f"{name}={value} must be divisible by block={block}")


def pack_a_mmt4d(a: Tensor) -> Tensor:
    m, k = a.shape
    _check_divisible("M", m, BLOCK_M)
    _check_divisible("K", k, BLOCK_K)
    return pack_a_mmt4d_kernel(
        a.view(m // BLOCK_M, BLOCK_M, k // BLOCK_K, BLOCK_K).contiguous()
    )


def pack_b_mmt4d(b: Tensor) -> Tensor:
    k, n = b.shape
    _check_divisible("K", k, BLOCK_K)
    _check_divisible("N", n, BLOCK_N)
    packed_panel = helion_block_pack.pack_b(b, BLOCK_N)
    return packed_panel.view(n // BLOCK_N, k // BLOCK_K, BLOCK_K, BLOCK_N)


def pack_bias(bias: Tensor) -> Tensor:
    n = bias.numel()
    _check_divisible("N", n, BLOCK_N)
    return bias.view(n // BLOCK_N, BLOCK_N).contiguous()


def view_merged_mmt4d(out4: Tensor) -> Tensor:
    blocks_m, block_m, blocks_n, block_n = out4.shape
    return out4.view(blocks_m * block_m, blocks_n * block_n)


def helion_linear(x: Tensor, layer: nn.Linear, relu: bool) -> Tensor:
    a4 = pack_a_mmt4d(x)
    weight_t = layer.weight.t().contiguous()
    b4 = pack_b_mmt4d(weight_t)
    bias2 = pack_bias(layer.bias)
    epilogue = relu_epilogue if relu else identity_epilogue
    out4 = linear_bf16_mmt4d_mlir(a4, b4, bias2, epilogue)
    return view_merged_mmt4d(out4)


def helion_mlp(x: Tensor, model: Model) -> Tensor:
    layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    hidden_layers = len(layers) - 1
    result = x
    for index, layer in enumerate(layers):
        result = helion_linear(result, layer, relu=index < hidden_layers)
    return result


def make_model(linear_layers: int) -> Model:
    if linear_layers == 1:
        hidden_layers: list[int] = []
    elif linear_layers == 3:
        hidden_layers = [HIDDEN_SIZE, HIDDEN_SIZE]
    else:
        raise ValueError("linear_layers must be 1 or 3")
    return Model(FEATURE_SIZE, hidden_layers, OUTPUT_SIZE).to(torch.bfloat16).eval()


def benchmark(name: str, operation: Callable[[], object]) -> float:
    for _ in range(WARMUP_ITERS):
        operation()

    timings_ms = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(BENCHMARK_ITERS):
            operation()
        timings_ms.append((time.perf_counter() - start) * 1_000 / BENCHMARK_ITERS)

    median_ms = statistics.median(timings_ms)
    print(f"{name:24s} {median_ms:8.3f} ms")
    return median_ms


def check_close(name: str, actual: Tensor, expected: Tensor) -> None:
    assert actual.dtype == expected.dtype
    abs_err = (actual.float() - expected.float()).abs()
    exact_mismatches = (actual != expected).sum().item()
    print(
        f"{name:24s} dtype {actual.dtype}, exact mismatches {exact_mismatches}, "
        f"max {abs_err.max().item():.3e}, mean {abs_err.mean().item():.3e}"
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1.0)


def run_case(linear_layers: int) -> None:
    model = make_model(linear_layers)
    x = torch.randn((BATCH_SIZE, FEATURE_SIZE), dtype=torch.float32).to(torch.bfloat16)

    with torch.inference_mode():
        expected = model(x)
        actual = helion_mlp(x, model)
        check_close(f"{linear_layers}-linear numerics", actual, expected)

        helion_ms = benchmark(
            f"Helion {linear_layers}-linear e2e",
            lambda: helion_mlp(x, model),
        )
        pytorch_ms = benchmark(
            f"PyTorch {linear_layers}-linear e2e",
            lambda: model(x),
        )

    layer_dims = [FEATURE_SIZE]
    layer_dims.extend(HIDDEN_SIZE for _ in range(max(0, linear_layers - 1)))
    layer_dims[-1:] = [OUTPUT_SIZE] if linear_layers == 1 else layer_dims[-1:]
    if linear_layers == 3:
        flops = 2 * BATCH_SIZE * FEATURE_SIZE * HIDDEN_SIZE
        flops += 2 * BATCH_SIZE * HIDDEN_SIZE * HIDDEN_SIZE
        flops += 2 * BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE
    else:
        flops = 2 * BATCH_SIZE * FEATURE_SIZE * OUTPUT_SIZE
    print(f"Helion {linear_layers}-linear    {flops / (helion_ms * 1e6):8.1f} GFLOP/s")
    print(f"PyTorch {linear_layers}-linear   {flops / (pytorch_ms * 1e6):8.1f} GFLOP/s")
    print(f"Helion/PyTorch {linear_layers}-linear {pytorch_ms / helion_ms:8.3f}x")


def main() -> None:
    if os.environ.get("HELION_MLIR_PIPELINE") != "1":
        raise RuntimeError("Set HELION_MLIR_PIPELINE=1 to use the vectorizing pipeline")
    for name, value, block in (
        ("BATCH", BATCH_SIZE, BLOCK_M),
        ("FEATURE", FEATURE_SIZE, BLOCK_K),
        ("HIDDEN", HIDDEN_SIZE, BLOCK_K),
        ("HIDDEN", HIDDEN_SIZE, BLOCK_N),
        ("OUTPUT", OUTPUT_SIZE, BLOCK_N),
    ):
        _check_divisible(name, value, block)

    threads = int(os.environ.get("OMP_NUM_THREADS", "64"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    print(
        f"bf16 MLP batch={BATCH_SIZE}, features={FEATURE_SIZE}, "
        f"hidden={HIDDEN_SIZE}, output={OUTPUT_SIZE}, blocks={BLOCK_M}x{BLOCK_N}x{BLOCK_K}, "
        f"threads={threads}"
    )
    run_case(1)
    run_case(3)


if __name__ == "__main__":
    main()
