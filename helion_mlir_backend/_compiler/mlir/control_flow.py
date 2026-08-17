"""Outer parallel control-flow lowering for MLIR kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlir.ir as ir
    import torch


def build_kernel_body(builder: object, out_tensor: torch.Tensor) -> ir.Value:
    """Build the outer ``scf.forall`` and its parallel insert terminator."""
    from mlir.dialects import scf as scf_d
    from mlir.dialects import tensor as tensor_d
    import mlir.ir as ir

    from .type_utils import torch_dtype_to_mlir

    grid_block_ids: list[int] = []
    for ids in builder.hf.device_ir.grid_block_ids:
        grid_block_ids.extend(ids)

    out_shape = [int(dim) for dim in out_tensor.shape]
    lbs = [0] * len(grid_block_ids)
    ubs = [out_shape[index] for index in range(len(grid_block_ids))]
    steps = [builder._block_id_to_size[block_id] for block_id in grid_block_ids]

    for block_id, upper_bound in zip(grid_block_ids, ubs, strict=False):
        previous = builder._block_id_to_upper_bound.get(block_id)
        if previous is None:
            builder._block_id_to_upper_bound[block_id] = int(upper_bound)
        else:
            builder._block_id_to_upper_bound[block_id] = min(previous, int(upper_bound))

    output_empty = tensor_d.EmptyOp(
        out_shape,
        torch_dtype_to_mlir(out_tensor.dtype),
    ).result
    forall = scf_d.ForallOp(lbs, ubs, steps, shared_outs=[output_empty])

    for block_id, induction_variable in zip(
        grid_block_ids,
        forall.induction_variables,
        strict=True,
    ):
        builder._block_id_to_iv[block_id] = induction_variable

    with ir.InsertionPoint(forall.body):
        shared_out = next(iter(forall.inner_iter_args))
        builder._process_root_graphs(shared_out)
        in_parallel = scf_d.InParallelOp()
        with ir.InsertionPoint(in_parallel.block):
            for value, offsets in builder._forall_insert_slices:
                source_type = ir.RankedTensorType(value.type)
                rank = len(source_type.shape)
                tensor_d.ParallelInsertSliceOp(
                    value,
                    shared_out,
                    offsets,
                    [],
                    [],
                    static_offsets=[ir.ShapedType.get_dynamic_size()] * rank,
                    static_sizes=list(source_type.shape),
                    static_strides=[1] * rank,
                )

    return forall.results[0]
