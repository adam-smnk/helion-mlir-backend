# Backend Shape Inference and Propagation (MLIR Backend)

How the MLIR backend decides tensor shapes.

Files: `analysis/geometry.py`, `support/index_meta.py`, `lowering/slice_plan.py`,
`build_context.py` (`shape_from_nodes`) and `aten_bridge/helpers.py`, all under
`helion_mlir_backend/_compiler/mlir/`.

## Principle

A shape is decided once, where the IR that needs it is built; everything
downstream reads it from MLIR types. Helion's node metadata (`meta["val"]`) is
read for dtypes, ranks and symbol origins. It is never modified, and never used
as the shape of emitted IR.

## Where Shapes Come From

1. **Host tensors**: the kernel's arguments have static shapes
   (`static_shapes=True`), which become the function argument types.
2. **Tiles**: `KernelGeometry` takes each block id's block size from the config
   and its loop span from the loop bounds. A tile's extent is
   `min(block size, span)` (`tile_extent`).
3. **Loads and stores**: `plan_slice` maps each index to a dimension slice
   (tile, scalar position, static slice, or gather) and gives static slice sizes.
   Block ids come from `resolve_index_descriptor`, which reads Helion's
   `meta['tile_with_offset']`, symbol origins (`HostFunction.expr_to_origin`),
   `_get_symnode('block_size_N')` keys and the symbol of a `sym_size.int`
   operand's dimension.
4. **Creation ops** (`hl.zeros`, `hl.full`): `shape_from_nodes` accepts ints,
   block sizes (as tile extents) and values that lowered to constants. Anything
   else raises a `DynamicShapeError`.
5. **Everything else** follows from the operands' MLIR types:
   - ATen helpers: running the op on meta tensors of the operand types gives the
     result types (`infer_results`); the helper signature is the call site's.
   - `view`/`reshape`: the result shape from `infer_results`.
   - `sym_size.int`: the operand's static MLIR dimension.
   - Contractions: `linalg` ops over the operand types.
   - Loop-carried values: `scf.for` iter args take their initial value's type.

## Failure Modes

- A torch-mlir helper whose lowered signature differs from the call site's
  raises an `UnsupportedOperationError` at the node's source line.
- Ragged tiles have extent `min(block size, span)`, where Helion expects a full
  masked block (see `docs/MLIR_LIMITATIONS.md`, sections 6 and 9).
- A shape expression without a static value raises a `DynamicShapeError`.

## Debugging

- `HELION_MLIR_DUMP_PRE_LOWERING=1` dumps the module before lowering. Helpers
  are the private `_aten_<op>_<hash>` functions, with their signatures.
- For an index question, call `resolve_index_descriptor(ctx, node)` directly;
  `tests/test_index_descriptor.py` covers the historical indexing patterns.
