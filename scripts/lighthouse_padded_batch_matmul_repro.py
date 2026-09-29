"""Minimal lighthouse reproducer: the optimizing pipeline miscompiles a
``linalg.batch_matmul`` whose operands are ``tensor.pad``-ed partial tiles (a ragged
reduction dim): the result contains NaN/garbage, while the scalar pipeline is exact.
The same padded tiles through a 2-D ``linalg.matmul`` are correct.

    uv run python scripts/lighthouse_padded_batch_matmul_repro.py [batch|plain]
"""

from __future__ import annotations

import sys

from mlir import ir
import torch

from helion_mlir_backend._compiler.execution import compile_entry
from helion_mlir_backend._compiler.mlir.codegen import _get_shared_mlir_context

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "batch"
B = "1x" if VARIANT == "batch" else ""
OP = "linalg.batch_matmul" if VARIANT == "batch" else "linalg.matmul"
ZERO = "0, " if VARIANT == "batch" else ""
ONE = "1, " if VARIANT == "batch" else ""
HIGH = "0, " if VARIANT == "batch" else ""
PAYLOAD = f"""
#min = affine_map<(d0) -> (-d0 + 40, 32)>
#rest = affine_map<(d0) -> (-d0 + 32)>
func.func public @k(
    %out: memref<{B}32x32xf32> {{helion.name = "out", helion.role = "inout"}},
    %xm: memref<{B}32x40xf32> {{helion.name = "x", helion.role = "in"}},
    %ym: memref<{B}40x32xf32> {{helion.name = "y", helion.role = "in"}}) {{
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c40 = arith.constant 40 : index
  %zero = arith.constant 0.0 : f32
  %x = bufferization.to_tensor %xm restrict : memref<{B}32x40xf32> to tensor<{B}32x40xf32>
  %y = bufferization.to_tensor %ym restrict : memref<{B}40x32xf32> to tensor<{B}40x32xf32>
  %e = tensor.empty() : tensor<{B}32x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<{B}32x32xf32>) -> tensor<{B}32x32xf32>
  %r = scf.for %k = %c0 to %c40 step %c32 iter_args(%acc = %init) -> (tensor<{B}32x32xf32>) {{
    %size = affine.min #min(%k)
    %pad = affine.apply #rest(%size)
    %xs = tensor.extract_slice %x[{ZERO}0, %k] [{ONE}32, %size] [{ONE}1, 1]
        : tensor<{B}32x40xf32> to tensor<{B}32x?xf32>
    %xp = tensor.pad %xs low[{ZERO}0, 0] high[{HIGH}0, %pad] {{
    ^bb0({"%b: index, " if B else ""}%i: index, %j: index):
      tensor.yield %zero : f32
    }} : tensor<{B}32x?xf32> to tensor<{B}32x32xf32>
    %ys = tensor.extract_slice %y[{ZERO}%k, 0] [{ONE}%size, 32] [{ONE}1, 1]
        : tensor<{B}40x32xf32> to tensor<{B}?x32xf32>
    %yp = tensor.pad %ys low[{ZERO}0, 0] high[{HIGH}%pad, 0] {{
    ^bb0({"%b: index, " if B else ""}%i: index, %j: index):
      tensor.yield %zero : f32
    }} : tensor<{B}?x32xf32> to tensor<{B}32x32xf32>
    %mm = {OP} ins(%xp, %yp : tensor<{B}32x32xf32>, tensor<{B}32x32xf32>)
        outs(%acc : tensor<{B}32x32xf32>) -> tensor<{B}32x32xf32>
    scf.yield %mm : tensor<{B}32x32xf32>
  }}
  bufferization.materialize_in_destination %r in restrict writable %out
      : (tensor<{B}32x32xf32>, memref<{B}32x32xf32>) -> ()
  return
}}
"""

torch.manual_seed(0)
shape = (1,) if VARIANT == "batch" else ()
x, y = torch.randn(*shape, 32, 40), torch.randn(*shape, 40, 32)
for pipeline in ("scalar", "opt"):
    with ir.Location.unknown(_get_shared_mlir_context()):
        module = ir.Module.parse(PAYLOAD)
        entry = compile_entry(module, "k", pipeline=pipeline)
    out = torch.zeros(*shape, 32, 32)
    entry([out, x, y])
    error = (out - x @ y).abs()
    print(
        f"{VARIANT} {pipeline}: nan={int(error.isnan().sum())} "
        f"max_err={error.nan_to_num(float('inf')).max().item():.3g}"
    )
