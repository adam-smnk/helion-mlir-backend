"""Minimal lighthouse reproducer: tile-and-fuse aborts when every dim of a fusion root is
smaller than the cache tile size (32), because all of its tile sizes become 0.

Expected with N < 32 on lighthouse without the ``get_fusion_roots`` all-zero skip:
assertion in LinalgTransformOps.cpp applyTilingToAll ("Mismatched number of loops").

    uv run python scripts/lighthouse_small_tile_repro.py 16
"""

from __future__ import annotations

import os
import sys

from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver
from mlir import ir

import helion_mlir_backend._compiler.execution as hm_exec

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PAYLOAD = f"""
func.func @k(%x: tensor<{N}xf32>) -> tensor<{N}xf32> {{
  %e = tensor.empty() : tensor<{N}xf32>
  %r = linalg.elementwise <add> ins(%x, %x : tensor<{N}xf32>, tensor<{N}xf32>) outs(%e : tensor<{N}xf32>) -> tensor<{N}xf32>
  return %r : tensor<{N}xf32>
}}
"""

context = ir.Context()
module = ir.Module.parse(PAYLOAD, context)
with context, ir.Location.unknown():
    driver = BackendDriver(module, "k", result_to_args=True, benchmark=False)
    driver.add_stage(
        Descriptor("./pipeline.yaml", base_path=os.path.dirname(hm_exec.__file__))
    )
    driver.apply(module)
print(f"N={N}: pipeline completed")
