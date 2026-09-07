"""BF16 MLP benchmark for Helion's MLIR backend.

The Helion path uses the mmt4d-style AMX matmul shape from
``helion_matmul_bf16.py``. Three properties keep per-forward work close to the
raw matmul cost:

* Weights are packed into ``[N/BN, K/BK, BK, BN]`` straight from the ``[N, K]``
  ``nn.Linear`` layout, so no eager transpose is ever needed.
* A hidden layer emits ``[M/BM, N/BN, BM, BN]``, which is exactly the packed-A
  layout the next layer consumes, so activations are packed once at the input
  and never re-packed between layers.
* Bias and the activation function are fused into the accumulator before the
  bf16 store.

Only end-to-end forward latency is reported. The headline Helion number packs
weights inside the timed region, because eager PyTorch is also handed plain
``[N, K]`` weights on every call. The `cached` row instead reuses weights packed
once; it models an inference runtime that caches them and is not comparable to
eager.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from typing import TYPE_CHECKING
from typing import NamedTuple

import helion
import helion.language as hl
import helion_numerics
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


class PackedLinear(NamedTuple):
    """One ``nn.Linear`` with weight and bias already in AMX block layout."""

    weight4: Tensor
    bias3: Tensor


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1, 8, 32]),
)
def pack_a_mmt4d_kernel(a4_src: Tensor) -> Tensor:
    """``[M/BM, BM, K/BK, BK]`` -> ``[M/BM, K/BK, BM, BK]``."""
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
    config=helion.Config(block_sizes=[1, 1, 32, 32]),
)
def pack_b_from_weight_kernel(w4_src: Tensor) -> Tensor:
    """``[N/BN, BN, K/BK, BK]`` -> ``[N/BN, K/BK, BK, BN]``."""
    blocks_n, block_n, blocks_k, block_k = w4_src.shape
    out = torch.empty(
        (blocks_n, blocks_k, block_k, block_n),
        dtype=w4_src.dtype,
        device=w4_src.device,
    )
    for block_ni, block_ki, tile_k, tile_n in hl.tile(
        [blocks_n, blocks_k, block_k, block_n]
    ):
        out[block_ni, block_ki, tile_k, tile_n] = w4_src[
            block_ni, tile_n, block_ki, tile_k
        ].permute(0, 2, 3, 1)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def linear_bf16_blocked_mlir(
    a4: Tensor,
    b4: Tensor,
    bias3: Tensor,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    """Linear emitting ``[M/BM, N/BN, BM, BN]``: the next layer's packed A."""
    blocks_m, blocks_k, block_m, block_k = a4.shape
    blocks_n, blocks_k2, block_k2, block_n = b4.shape
    assert blocks_k == blocks_k2, "major K mismatch"
    assert block_k == block_k2, "minor K mismatch"

    out = torch.empty(
        (blocks_m, blocks_n, block_m, block_n),
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
        y = epilogue(acc + bias3[tile_blocks_n, :, :])
        out[tile_blocks_m, tile_blocks_n, :, :] = y.to(a4.dtype)
    return out


@helion.kernel(
    static_shapes=True,
    backend="mlir",
    config=helion.Config(block_sizes=[1, 1]),
)
def linear_bf16_merged_mlir(
    a4: Tensor,
    b4: Tensor,
    bias3: Tensor,
    epilogue: Callable[[Tensor], Tensor],
) -> Tensor:
    """Linear emitting ``[M/BM, BM, N/BN, BN]``, viewable as row-major ``[M, N]``."""
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
        y = epilogue(acc + bias3[tile_blocks_n, :, :])
        out[tile_blocks_m, :, tile_blocks_n, :] = y.to(a4.dtype).permute(0, 2, 1, 3)
    return out


def identity_epilogue(x: Tensor) -> Tensor:
    return x


def relu_epilogue(x: Tensor) -> Tensor:
    return torch.relu(x)


def _check_divisible(name: str, value: int, block: int) -> None:
    if value % block:
        raise ValueError(f"{name}={value} must be divisible by block={block}")


def pack_a_mmt4d(a: Tensor) -> Tensor:
    """Pack row-major activations ``[M, K]`` into ``[M/BM, K/BK, BM, BK]``."""
    m, k = a.shape
    _check_divisible("M", m, BLOCK_M)
    _check_divisible("K", k, BLOCK_K)
    return pack_a_mmt4d_kernel(
        a.view(m // BLOCK_M, BLOCK_M, k // BLOCK_K, BLOCK_K).contiguous()
    )


def pack_linear(layer: nn.Linear) -> PackedLinear:
    """Pack one ``nn.Linear``; ``weight`` is ``[N, K]`` so no transpose is needed."""
    n, k = layer.weight.shape
    _check_divisible("N", n, BLOCK_N)
    _check_divisible("K", k, BLOCK_K)
    weight4 = pack_b_from_weight_kernel(
        layer.weight.view(n // BLOCK_N, BLOCK_N, k // BLOCK_K, BLOCK_K).contiguous()
    )
    bias3 = layer.bias.view(n // BLOCK_N, 1, BLOCK_N).contiguous()
    return PackedLinear(weight4=weight4, bias3=bias3)


def pack_model(model: Model) -> list[PackedLinear]:
    return [
        pack_linear(layer) for layer in model.network if isinstance(layer, nn.Linear)
    ]


def view_merged_mmt4d(out4: Tensor) -> Tensor:
    blocks_m, block_m, blocks_n, block_n = out4.shape
    return out4.view(blocks_m * block_m, blocks_n * block_n)


def helion_mlp(x: Tensor, packed: list[PackedLinear]) -> Tensor:
    """Forward over already-packed layers; activations stay blocked between layers."""
    a4 = pack_a_mmt4d(x)
    for layer in packed[:-1]:
        a4 = linear_bf16_blocked_mlir(a4, layer.weight4, layer.bias3, relu_epilogue)
    last = packed[-1]
    return view_merged_mmt4d(
        linear_bf16_merged_mlir(a4, last.weight4, last.bias3, identity_epilogue)
    )


def helion_mlp_online(x: Tensor, model: Model) -> Tensor:
    """Forward that also packs weights, matching eager PyTorch's per-call work."""
    return helion_mlp(x, pack_model(model))


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


def reference_f64(x: Tensor, model: Model) -> Tensor:
    """Ground truth with bf16 rounding only where both implementations store it."""
    layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    out = x.double()
    for index, layer in enumerate(layers):
        out = out @ layer.weight.double().t() + layer.bias.double()
        if index < len(layers) - 1:
            out = torch.relu(out)
        out = out.to(torch.bfloat16).double()
    return out


def strict_small_shape_check(linear_layers: int) -> None:
    """Bit-exact gate on inputs whose exact result is representable in bf16.

    Signed-permutation weights keep every layer's magnitude bounded, so the
    result stays exact no matter how many layers are chained.
    """
    size = 256
    generator = torch.Generator().manual_seed(0)
    hidden = [size, size] if linear_layers == 3 else []
    model = Model(size, hidden, size).to(torch.bfloat16).eval()
    with torch.no_grad():
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                weight = helion_numerics.signed_permutation(size, generator)
                layer.weight.copy_(weight.to(torch.bfloat16))
                bias = torch.randint(-8, 9, (size,), generator=generator)
                layer.bias.copy_(bias.to(torch.bfloat16))
    x = helion_numerics.small_integer_matrix(size, size, generator, limit=8).to(
        torch.bfloat16
    )

    exact = reference_f64(x, model).to(torch.bfloat16)
    with torch.inference_mode():
        helion_numerics.assert_bitwise_equal(
            f"  {linear_layers}-linear helion", helion_mlp_online(x, model), exact
        )
        helion_numerics.assert_bitwise_equal(
            f"  {linear_layers}-linear torch", model(x), exact
        )


def run_case(linear_layers: int) -> None:
    model = make_model(linear_layers)
    x = torch.randn((BATCH_SIZE, FEATURE_SIZE), dtype=torch.float32).to(torch.bfloat16)

    with torch.inference_mode():
        packed = pack_model(model)

        expected = model(x)
        actual = helion_mlp_online(x, model)
        truth = reference_f64(x, model)
        check_close(f"{linear_layers}-linear numerics", actual, expected)
        helion_numerics.assert_no_worse_than(
            f"{linear_layers}-linear vs f64", actual, expected, truth
        )

        # Comparable with eager PyTorch: weights arrive in plain [N, K] layout.
        helion_ms = benchmark(
            f"Helion {linear_layers}-linear e2e",
            lambda: helion_mlp_online(x, model),
        )
        pytorch_ms = benchmark(
            f"PyTorch {linear_layers}-linear e2e",
            lambda: model(x),
        )
        # Not comparable with eager: models an inference runtime caching weights.
        cached_ms = benchmark(
            f"Helion {linear_layers}-lin cached",
            lambda: helion_mlp(x, packed),
        )

    if linear_layers == 3:
        flops = 2 * BATCH_SIZE * FEATURE_SIZE * HIDDEN_SIZE
        flops += 2 * BATCH_SIZE * HIDDEN_SIZE * HIDDEN_SIZE
        flops += 2 * BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE
    else:
        flops = 2 * BATCH_SIZE * FEATURE_SIZE * OUTPUT_SIZE
    print(f"Helion {linear_layers}-linear    {flops / (helion_ms * 1e6):8.1f} GFLOP/s")
    print(f"PyTorch {linear_layers}-linear   {flops / (pytorch_ms * 1e6):8.1f} GFLOP/s")
    print(f"Helion/PyTorch {linear_layers}-linear {pytorch_ms / helion_ms:8.3f}x")
    print(
        f"Helion {linear_layers}-lin cached  {flops / (cached_ms * 1e6):8.1f} GFLOP/s"
        f" (weights pre-packed, not eager-comparable)"
    )


def main() -> None:
    if os.environ.get("HELION_MLIR_PIPELINE") != "1":
        raise RuntimeError("Set HELION_MLIR_PIPELINE=1 to use the vectorizing pipeline")
    if BLOCK_N != BLOCK_K:
        raise ValueError(
            "BLOCK_N must equal BLOCK_K so a hidden layer's blocked output is a"
            " valid packed A for the next layer"
        )
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
    print("strict bit-exact check at 256x256 (signed-permutation weights)")
    strict_small_shape_check(1)
    strict_small_shape_check(3)
    run_case(1)
    run_case(3)


if __name__ == "__main__":
    main()
