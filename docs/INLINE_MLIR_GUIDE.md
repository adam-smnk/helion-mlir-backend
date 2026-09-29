# Inline MLIR Guide

`helion_mlir_backend.inline_mlir` calls a hand-written MLIR function on tiles inside
a Helion device loop. Use it for code that Helion's PyTorch-level operations cannot
express, or cannot express efficiently: a custom elementwise formula, a
hand-scheduled microkernel, target-specific operations such as the `x86` dialect.

The function is cloned into the kernel's module and inlined into the kernel before
bufferization, so the lighthouse pipeline lowers it together with the rest of the
kernel. It is not a black box: the pipeline may transform its operations like any
others.

See [examples/inline_mlir.py](../examples/inline_mlir.py) for a `linalg.generic`
snippet, an `ir.Module` built with the Python bindings, and a memref/vector matmul
microkernel compared with an identity-layout variant after bufferization.

## Quick start

```python
import helion
import helion.language as hl
import torch

from helion_mlir_backend import inline_mlir

SOFTPLUS = """
#id = affine_map<(i, j) -> (i, j)>
func.func @softplus(%x: tensor<?x?xf32>, %beta: f32) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%v: f32, %unused: f32):
    ...
    linalg.yield %r : f32
  } -> tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}
"""


def softplus_reference(x, beta):
    return torch.nn.functional.softplus(x, beta)


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 32]))
def softplus(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        t = x[tile]
        out[tile] = inline_mlir(SOFTPLUS, [t, 2.0], t, reference=softplus_reference)
    return out
```

## The call

```python
inline_mlir(source, args, output_like, *, reference=None)
```

| Argument | Meaning |
|---|---|
| `source` | MLIR text, or an `mlir.ir.Module`, holding only `func.func` ops. The first one is the entry and is called; the others are its helpers or external declarations. |
| `args` | The entry's arguments, in order. |
| `output_like` | A tensor, or a tuple of tensors: the shape and dtype of each result as the kernel sees it. Usually a tile of the same shape, e.g. the input tile. |
| `reference` | Optional Python function of `args` computing the same results; see [Reference implementations](#reference-implementations). |

It returns the result tensor, or a tuple of them when `output_like` is a sequence.

`source` must be a module-level global (or a literal): Helion kernels cannot capture
local variables. A module can come from any `ir.Context`, parsed or built with the
Python bindings; it is printed with its locations and parsed again in the backend's
context. It is read when the kernel is first compiled; later changes to it are not
seen.

### Arguments and parameters

The entry's parameters and results are ranked tensors, or `index`, integer and float
scalars. Memrefs are not allowed at the boundary (see [Memrefs in the
body](#memrefs-in-the-body)); results must be tensors.

| Kernel argument | Parameter type | Conversion |
|---|---|---|
| Tile or other device tensor (`x[tile]`, `hl.zeros(...)`, a computed tile) | `tensor<...xT>` of the same rank and element type | `tensor.cast` when the shapes differ only in `?` dims |
| Host tensor slice (`table[:, tn]`) | `tensor<...xT>` | as above; read-only |
| `tile.begin`, `tile.end`, sizes | `index` or an integer | `arith.index_cast` |
| Kernel `int`/`float` parameter | integer or float | cast with torch semantics |
| Python number | integer or float (floats only for float parameters) | `arith.constant` of the parameter type |

Results are cast from the declared result types to the types of `output_like`.

## What the snippet sees

- **Whole blocks.** A tile loaded with `x[tile]` spans the block size, clipped to
  the dimension: `min(block_size, dim)`. At the edge of a tensor it is padded with
  zeros up to that extent; results past the edge are dropped by the store. Code
  that must not see the padding (a `max`, a reduction) can take `tile.begin` and
  `tile.end` as arguments and mask itself.
- **Full dimensions.** `x[tm, :]` spans the whole dimension.
- **Shapes.** With `static_shapes=True` (the default) tiles have static shapes;
  with dynamic shapes some dims are `?`.

Declare `?` for dims that depend on the config or on the input shapes: the snippet
then works for every block size the autotuner may try. Static dims are fine when you
fix the config, e.g. for a microkernel written for 8x16 tiles. They must equal the
kernel's static extent; a mismatch is an error when the kernel is compiled. With
dynamic shapes, a static dim receiving a `?` dim is a promise that the extents match
at run time.

## Semantics

- The call is a pure function of its arguments with value semantics, like any
  tensor operation. If its results are unused, it is removed.
- It runs once per tile per iteration of the enclosing loop.
- After inlining, its operations are part of the kernel: they are canonicalized,
  bufferized and lowered with it, and the opt pipeline's transforms may apply to
  them (e.g. tile and vectorize a `linalg.matmul` in the snippet).

## Validation

1. **While Helion traces the kernel** (the first call, before any MLIR is
   generated). Errors point at the `inline_mlir(...)` line:
   - the source parses and verifies;
   - it holds `func.func` ops only, and the entry has a body;
   - parameters and results have allowed types (no memrefs), results are tensors;
   - argument and result counts match the call;
   - tensor arguments and `output_like` match in rank and element type; scalars
     are not passed for tensors, floats not for integers.

   Tile sizes are still symbolic here, so static dims are checked in the next step.
2. **When MLIR is generated**, with the config known: static dims against the
   kernel's tile types.
3. **In the lighthouse pipeline**: anything that verifies but cannot be lowered
   (e.g. an op the pipeline does not convert, or elementwise `arith`/`math` ops on
   tensors, which bufferization does not handle; use `linalg`). Locations
   `"-":line:col` refer to lines of the snippet text.

The first step also reports advice as an `InlineMLIRHint` warning, once per snippet;
silence it with `@helion.kernel(ignore_warnings=[helion_mlir_backend.InlineMLIRHint])`.

## Memrefs in the body

The entry's boundary is tensors, but its body may use memrefs, e.g. for a
hand-written microkernel with operations that only take memrefs or vectors.
Convert with `bufferization.to_buffer` and back with `bufferization.to_tensor`.
These follow upstream One-Shot Bufferization rules; the points that matter here:

**Read-only inputs.** `bufferization.to_buffer %x read_only` gives the buffer
behind `%x` without a copy. `read_only` is a promise: writing through that buffer
(or a view of it) writes the caller's data, e.g. the kernel's input tensor. Without
`read_only`, writing is safe: bufferization copies the tile first when its data is
still needed elsewhere.

**Layouts.** Tiles are strided views of the kernel's tensors, often with strides
known only at run time. `to_buffer` must produce a type that bufferization can give
without a copy, so declare a fully dynamic layout and cast it inside the snippet to
the layout your code needs:

```mlir
%any = bufferization.to_buffer %a read_only
    : tensor<8x8xf32> to memref<8x8xf32, strided<[?, ?], offset: ?>>
%A = memref.cast %any
    : memref<8x8xf32, strided<[?, ?], offset: ?>> to memref<8x8xf32, strided<[?, 1], offset: ?>>
```

An identity layout (`memref<8x8xf32>`) makes bufferization copy the tile on every
call: in the example's matmul microkernel, it adds an allocation and a copy of the
`B` tile to every step of the K loop, which the fully dynamic layout avoids. A
partially static layout (`strided<[?, 1], offset: ?>`) directly on
`to_buffer` can make it allocate a buffer of that layout, which fails with
`'memref.alloc' op symbol operand count does not equal memref symbol count`. The
`InlineMLIRHint` warning points out `to_buffer` of an argument without a fully
dynamic layout.

**Results.** Return `bufferization.to_tensor %buffer restrict writable`. `restrict`
is required by bufferization and is again a promise: no other tensor is created
from that buffer and it is not accessed after the conversion.

**Allocations.** `memref.alloc` is freed by the pipeline's deallocation pass.
`memref.alloca` takes stack space on every call, released only when the kernel
returns: in a loop over many tiles it overflows the stack (4096x4096 with 8x16
tiles overflowed an 8 MB stack). Use it only for scratch inside
`memref.alloca_scope`.

**Writing results in place.** Converting between tensors and memrefs hides the data
flow from bufferization, so a result built in a buffer the snippet allocates costs
one allocation and one copy per tile into the output. To write straight into the
output, pass the destination as an argument, write through a plain `to_buffer` of
it, and return its `to_tensor`:

```python
out[tm, tn] = inline_mlir(SCALE_INTO, [x[tm, tn], out[tm, tn]], x[tm, tn])
```

| Where the snippet writes (per full tile) | Allocations / copies |
|---|---|
| A buffer it allocates | 1 / 1 |
| A fresh destination (`hl.zeros([tm, tn])`) | 1 / 1 |
| The output's own tile (`out[tm, tn]`), stored back to `out[tm, tn]` | 0 / 0 |

Edge tiles are padded copies and always allocate. The kernel reads `out` to pass
it, which is harmless when the snippet overwrites it.

**Accumulators.** An accumulator passed in and returned through the same buffer
(`acc = inline_mlir(..., [acc, a_tile, b_tile], acc)` in a reduction loop) is
updated in place across iterations; see the microkernel in the example.

## Dialects

The snippet may use any dialect registered in the MLIR Python bindings, as long as
the pipeline can lower it: `arith`, `math`, `linalg`, `tensor`, `scf`, `affine`,
`vector`, `memref`, `bufferization`, and the `x86` dialect (the former `x86vector`
and `amx`, e.g. `x86.avx.rsqrt`, `x86.amx.tile_load`). An `x86.avx.rsqrt`
microkernel runs with both pipelines; target-specific operations need a machine
that supports them. Custom dialects are not available. Both pipelines lower
`linalg.pack` and `linalg.unpack` (see the inline `linalg.pack` in
`examples/block_packing_mlir.py`).

## Reference implementations

`reference` is called with the same `args` instead of the snippet:

- in Helion's eager ref mode (`ref_mode=helion.RefMode.EAGER`), with real tensors;
- on backends other than MLIR, where Helion traces it like any device code, so it
  must be written with operations Helion supports (e.g. `torch` ops on tiles).

Without a `reference`, `inline_mlir` raises in those cases.

## Debugging

- `generate_mlir(kernel, args)` shows the module before inlining: the snippet's
  functions are private and named `__inline_mlir_<hash>_<name>`, and each call site
  is a `func.call` to the entry.
- `HELION_MLIR_DUMP_IR`, `HELION_MLIR_DUMP_PRE_LOWERING` (after inlining) and
  `HELION_MLIR_DUMP_LOWERED` print the module at each stage.
- A snippet can be tried on its own with `mlir-opt` or the Python bindings before
  being used in a kernel.

## Limitations

- Function form only: the entry is a complete `func.func`.
- Results are tensors; memrefs are not allowed in the entry's signature.
- Snippets are module globals, read at the first compile.
- A static-shaped snippet fails to compile for configs whose tiles have other
  extents; with autotuning, prefer `?` dims.
- Helion's autotuner does not know what the snippet computes; it must be correct
  for any block size the search may pick.
