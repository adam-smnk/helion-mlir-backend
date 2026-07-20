func.func @matmul(%arg0: tensor<128x1024xf32>, %arg1: tensor<1024x512xf32>) -> tensor<128x512xf32> {
  %out = tensor.empty() : tensor<128x512xf32>
  %1 = scf.forall (%tile_m_idx, %tile_n_idx) = (0, 0) to (128, 512) step (32, 32) shared_outs(%arg3 = %out) -> (tensor<128x512xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%tile_m_idx, 0] [32, 1024] [1, 1] : tensor<128x1024xf32> to tensor<32x1024xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %tile_n_idx] [1024, 32] [1, 1] : tensor<1024x512xf32> to tensor<1024x32xf32>
    %acc_tile = tensor.extract_slice %arg3[%tile_m_idx, %tile_n_idx] [32, 32] [1, 1] : tensor<128x512xf32> to tensor<32x32xf32>
    
    %zero = arith.constant 0.0 : f32
    %acc = linalg.fill ins(%zero : f32) outs(%acc_tile : tensor<32x32xf32>) -> tensor<32x32xf32>
    
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %res = scf.for %arg5 = %c0 to %c1024 step %c64 iter_args(%arg4 = %acc) -> (tensor<32x32xf32>) {
      %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg5] [32, 64] [1, 1] : tensor<32x1024xf32> to tensor<32x64xf32>
      %extracted_slice_3 = tensor.extract_slice %extracted_slice_1[%arg5, 0] [64, 32] [1, 1] : tensor<1024x32xf32> to tensor<64x32xf32>
      %1 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%arg4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.yield %1 : tensor<32x32xf32>
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %res into %arg3[%tile_m_idx, %tile_n_idx] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x512xf32>
    }
  }
  return %1 : tensor<128x512xf32>
}
