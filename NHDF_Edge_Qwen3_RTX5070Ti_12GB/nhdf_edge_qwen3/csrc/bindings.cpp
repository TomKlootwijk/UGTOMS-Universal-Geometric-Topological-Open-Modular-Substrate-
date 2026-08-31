#include <torch/extension.h>


torch::Tensor nhdf_gemv_cuda(
    torch::Tensor x,
    torch::Tensor base_codes,
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor residual_mask_words,
    torch::Tensor residual_prefix,
    torch::Tensor residual_bits,
    torch::Tensor residual_scales,
    int64_t rows,
    int64_t original_cols,
    int64_t padded_cols,
    int64_t groups_per_row,
    int64_t bits,
    int64_t group_size,
    int64_t row_offset,
    int64_t row_count);


torch::Tensor nhdf_dequantize_rows_cuda(
    torch::Tensor row_indices,
    torch::Tensor base_codes,
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor residual_mask_words,
    torch::Tensor residual_prefix,
    torch::Tensor residual_bits,
    torch::Tensor residual_scales,
    int64_t rows,
    int64_t original_cols,
    int64_t padded_cols,
    int64_t groups_per_row,
    int64_t bits,
    int64_t group_size);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &nhdf_gemv_cuda, "NHDF fused packed GEMV (CUDA)");
    m.def("dequantize_rows", &nhdf_dequantize_rows_cuda,
          "NHDF selected-row dequantization (CUDA)");
}
