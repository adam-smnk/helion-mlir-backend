"""Linear helpers with optional deployment-style weight prepacking."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple

from .matmul import identity_epilogue
from .matmul import matmul
from .matmul import matmul_prepacked_b
from .matmul import matmul_prepacked_b_affine
from .matmul import pack_b_blocked_t

if TYPE_CHECKING:
    from torch import Tensor
    from torch import nn

CACHE_PREPACKED_WEIGHTS_ENV = "HELION_MLIR_CACHE_PREPACKED_WEIGHTS"


class LinearCache(NamedTuple):
    key: tuple
    packed_weight: Tensor
    bias: Tensor | None
    out_features: int


class AffineLinearCache(NamedTuple):
    key: tuple
    packed_weight: Tensor
    bias: Tensor | None
    post_scale: Tensor
    post_bias: Tensor | None
    out_features: int


def _parameter_key(parameter: Tensor | None) -> tuple | None:
    if parameter is None:
        return None
    return (
        parameter.data_ptr(),
        parameter._version,
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
    )


def _combine_biases(biases: tuple[Tensor, ...], x: Tensor) -> Tensor | None:
    if not biases:
        return None
    bias = biases[0].to(dtype=x.dtype, device=x.device)
    for extra_bias in biases[1:]:
        bias = bias + extra_bias.to(dtype=x.dtype, device=x.device)
    return bias


def linear(
    x: Tensor,
    layer: nn.Linear,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    cache: LinearCache | None = None,
    biases: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, LinearCache | None]:
    """Run a linear layer, optionally caching its packed constant weight.

    The default path packs the weight on every call. Set
    ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=1`` to build and reuse a packed
    weight until the source parameters, input dtype, or device change.
    """
    source_biases = (
        (() if layer.bias is None else (layer.bias,)) if biases is None else biases
    )
    if os.environ.get(CACHE_PREPACKED_WEIGHTS_ENV, "").strip() != "1":
        weight = layer.weight.to(dtype=x.dtype, device=x.device)
        return (
            matmul(
                x,
                weight,
                trans_b=True,
                bias=_combine_biases(source_biases, x),
                epilogue=epilogue,
            ),
            None,
        )

    key = (
        _parameter_key(layer.weight),
        tuple(_parameter_key(bias) for bias in source_biases),
        x.dtype,
        x.device,
    )
    if cache is None or cache.key != key:
        weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
        bias = _combine_biases(source_biases, x)
        cache = LinearCache(
            key,
            pack_b_blocked_t(weight),
            bias,
            int(layer.out_features),
        )

    return (
        matmul_prepacked_b(
            x,
            cache.packed_weight,
            n=cache.out_features,
            bias=cache.bias,
            epilogue=epilogue,
        ),
        cache,
    )


def linear_affine(
    x: Tensor,
    layer: nn.Linear,
    post_scale: Tensor,
    post_bias: Tensor | None = None,
    epilogue: Callable[[Tensor], Tensor] = identity_epilogue,
    cache: AffineLinearCache | None = None,
) -> tuple[Tensor, AffineLinearCache | None]:
    """Run ``epilogue(linear(x) * post_scale + post_bias)``."""
    key = (
        _parameter_key(layer.weight),
        _parameter_key(layer.bias),
        _parameter_key(post_scale),
        _parameter_key(post_bias),
        x.dtype,
        x.device,
    )
    use_cache = os.environ.get(CACHE_PREPACKED_WEIGHTS_ENV, "").strip() == "1"
    if not use_cache or cache is None or cache.key != key:
        weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
        bias = (
            None
            if layer.bias is None
            else layer.bias.detach().to(dtype=x.dtype, device=x.device)
        )
        cache = AffineLinearCache(
            key,
            pack_b_blocked_t(weight),
            bias,
            post_scale.detach().to(dtype=x.dtype, device=x.device),
            None
            if post_bias is None
            else post_bias.detach().to(dtype=x.dtype, device=x.device),
            int(layer.out_features),
        )

    result = matmul_prepacked_b_affine(
        x,
        cache.packed_weight,
        cache.out_features,
        cache.bias,
        cache.post_scale,
        cache.post_bias,
        epilogue,
    )
    return result, cache if use_cache else None
