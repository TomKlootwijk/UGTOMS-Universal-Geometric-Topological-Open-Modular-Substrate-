#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float level_value(int code, int bits) {
    return bits == 2 ? (static_cast<float>(code) - 1.5f)
                     : (static_cast<float>(code) - 7.5f);
}

__device__ __forceinline__ int decode_code(
    const uint8_t* __restrict__ packed,
    int64_t value_index,
    int bits) {
    if (bits == 2) {
        const uint8_t byte = packed[value_index >> 2];
        return (byte >> ((value_index & 3) * 2)) & 0x3;
    }
    const uint8_t byte = packed[value_index >> 1];
    return (byte >> ((value_index & 1) * 4)) & 0xF;
}

__device__ __forceinline__ int selected_rank(
    int64_t group_id,
    const int32_t* __restrict__ mask_words,
    const int32_t* __restrict__ prefix) {
    const int64_t word_index = group_id >> 5;
    const int bit_index = static_cast<int>(group_id & 31);
    const uint32_t word = static_cast<uint32_t>(mask_words[word_index]);
    if (((word >> bit_index) & 1u) == 0u) {
        return -1;
    }
    const uint32_t earlier = bit_index == 0 ? 0u : (word & ((1u << bit_index) - 1u));
    return prefix[word_index] + __popc(earlier);
}

template <typename scalar_t>
__global__ void gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ base_codes,
    const at::Half* __restrict__ means,
    const at::Half* __restrict__ scales,
    const int32_t* __restrict__ residual_mask_words,
    const int32_t* __restrict__ residual_prefix,
    const uint8_t* __restrict__ residual_bits,
    const at::Half* __restrict__ residual_scales,
    scalar_t* __restrict__ output,
    int64_t original_cols,
    int64_t padded_cols,
    int64_t groups_per_row,
    int bits,
    int group_size,
    int64_t row_offset,
    int64_t row_count) {
    const int64_t local_row = blockIdx.x;
    const int64_t batch = blockIdx.y;
    if (local_row >= row_count) {
        return;
    }
    const int64_t row = row_offset + local_row;
    float sum = 0.0f;

    for (int64_t col = threadIdx.x; col < original_cols; col += blockDim.x) {
        const int64_t value_index = row * padded_cols + col;
        const int code = decode_code(base_codes, value_index, bits);
        const int64_t group_id = row * groups_per_row + col / group_size;
        float weight = static_cast<float>(means[group_id])
                     + static_cast<float>(scales[group_id]) * level_value(code, bits);

        if (residual_mask_words != nullptr) {
            const int rank = selected_rank(group_id, residual_mask_words, residual_prefix);
            if (rank >= 0) {
                const int within_group = static_cast<int>(col % group_size);
                const int64_t residual_value = static_cast<int64_t>(rank) * group_size + within_group;
                const uint8_t byte = residual_bits[residual_value >> 3];
                const bool positive = ((byte >> (residual_value & 7)) & 1u) != 0u;
                const float sign = positive ? 1.0f : -1.0f;
                weight += sign * static_cast<float>(residual_scales[rank]);
            }
        }
        sum += static_cast<float>(x[batch * original_cols + col]) * weight;
    }

    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[batch * row_count + local_row] = static_cast<scalar_t>(scratch[0]);
    }
}

__global__ void dequantize_rows_kernel(
    const int64_t* __restrict__ row_indices,
    int64_t selected_rows,
    const uint8_t* __restrict__ base_codes,
    const at::Half* __restrict__ means,
    const at::Half* __restrict__ scales,
    const int32_t* __restrict__ residual_mask_words,
    const int32_t* __restrict__ residual_prefix,
    const uint8_t* __restrict__ residual_bits,
    const at::Half* __restrict__ residual_scales,
    at::Half* __restrict__ output,
    int64_t original_cols,
    int64_t padded_cols,
    int64_t groups_per_row,
    int bits,
    int group_size) {
    const int64_t selected_row = blockIdx.x;
    if (selected_row >= selected_rows) {
        return;
    }
    const int64_t row = row_indices[selected_row];
    for (int64_t col = threadIdx.x; col < original_cols; col += blockDim.x) {
        const int64_t value_index = row * padded_cols + col;
        const int code = decode_code(base_codes, value_index, bits);
        const int64_t group_id = row * groups_per_row + col / group_size;
        float weight = static_cast<float>(means[group_id])
                     + static_cast<float>(scales[group_id]) * level_value(code, bits);
        if (residual_mask_words != nullptr) {
            const int rank = selected_rank(group_id, residual_mask_words, residual_prefix);
            if (rank >= 0) {
                const int within_group = static_cast<int>(col % group_size);
                const int64_t residual_value = static_cast<int64_t>(rank) * group_size + within_group;
                const uint8_t byte = residual_bits[residual_value >> 3];
                const float sign = ((byte >> (residual_value & 7)) & 1u) ? 1.0f : -1.0f;
                weight += sign * static_cast<float>(residual_scales[rank]);
            }
        }
        output[selected_row * original_cols + col] = static_cast<at::Half>(weight);
    }
}

void check_common(
    const torch::Tensor& base_codes,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& residual_mask_words,
    const torch::Tensor& residual_prefix,
    const torch::Tensor& residual_bits,
    const torch::Tensor& residual_scales,
    int64_t rows,
    int64_t original_cols,
    int64_t padded_cols,
    int64_t groups_per_row,
    int64_t bits,
    int64_t group_size) {
    TORCH_CHECK(base_codes.is_cuda(), "base_codes must be CUDA");
    TORCH_CHECK(base_codes.scalar_type() == at::kByte, "base_codes must be uint8");
    TORCH_CHECK(means.is_cuda() && means.scalar_type() == at::kHalf, "means must be CUDA float16");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kHalf, "scales must be CUDA float16");
    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
    TORCH_CHECK(group_size > 0 && group_size % 256 == 0, "group_size must be a multiple of 256");
    TORCH_CHECK(padded_cols >= original_cols, "padded_cols must cover original_cols");
    TORCH_CHECK(groups_per_row * group_size == padded_cols, "inconsistent group geometry");
    TORCH_CHECK(means.numel() == rows * groups_per_row, "mean count mismatch");
    TORCH_CHECK(scales.numel() == rows * groups_per_row, "scale count mismatch");
    if (residual_mask_words.numel() > 0) {
        TORCH_CHECK(residual_mask_words.is_cuda() && residual_mask_words.scalar_type() == at::kInt,
                    "residual_mask_words must be CUDA int32");
        TORCH_CHECK(residual_prefix.is_cuda() && residual_prefix.scalar_type() == at::kInt,
                    "residual_prefix must be CUDA int32");
        TORCH_CHECK(residual_bits.is_cuda() && residual_bits.scalar_type() == at::kByte,
                    "residual_bits must be CUDA uint8");
        TORCH_CHECK(residual_scales.is_cuda() && residual_scales.scalar_type() == at::kHalf,
                    "residual_scales must be CUDA float16");
    }
}

}  // namespace


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
    int64_t row_count) {
    check_common(base_codes, means, scales, residual_mask_words, residual_prefix,
                 residual_bits, residual_scales, rows, original_cols, padded_cols,
                 groups_per_row, bits, group_size);
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [batch, in_features]");
    TORCH_CHECK(x.size(1) == original_cols, "x width mismatch");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat,
                "x must be float16 or float32");
    TORCH_CHECK(row_offset >= 0 && row_count >= 0 && row_offset + row_count <= rows,
                "row interval outside matrix");

    c10::cuda::CUDAGuard guard(x.device());
    auto output = torch::empty({x.size(0), row_count}, x.options());
    const dim3 grid(static_cast<unsigned int>(row_count), static_cast<unsigned int>(x.size(0)));
    const dim3 block(kThreads);
    const int32_t* mask_ptr = residual_mask_words.numel() ? residual_mask_words.data_ptr<int32_t>() : nullptr;
    const int32_t* prefix_ptr = residual_mask_words.numel() ? residual_prefix.data_ptr<int32_t>() : nullptr;
    const uint8_t* residual_bits_ptr = residual_mask_words.numel() ? residual_bits.data_ptr<uint8_t>() : nullptr;
    const at::Half* residual_scales_ptr = residual_mask_words.numel() ? residual_scales.data_ptr<at::Half>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == at::kHalf) {
        gemv_kernel<at::Half><<<grid, block, 0, stream>>>(
            x.data_ptr<at::Half>(), base_codes.data_ptr<uint8_t>(), means.data_ptr<at::Half>(),
            scales.data_ptr<at::Half>(), mask_ptr, prefix_ptr, residual_bits_ptr,
            residual_scales_ptr, output.data_ptr<at::Half>(), original_cols, padded_cols,
            groups_per_row, static_cast<int>(bits), static_cast<int>(group_size),
            row_offset, row_count);
    } else {
        gemv_kernel<float><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(), base_codes.data_ptr<uint8_t>(), means.data_ptr<at::Half>(),
            scales.data_ptr<at::Half>(), mask_ptr, prefix_ptr, residual_bits_ptr,
            residual_scales_ptr, output.data_ptr<float>(), original_cols, padded_cols,
            groups_per_row, static_cast<int>(bits), static_cast<int>(group_size),
            row_offset, row_count);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


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
    int64_t group_size) {
    check_common(base_codes, means, scales, residual_mask_words, residual_prefix,
                 residual_bits, residual_scales, rows, original_cols, padded_cols,
                 groups_per_row, bits, group_size);
    TORCH_CHECK(row_indices.is_cuda() && row_indices.scalar_type() == at::kLong,
                "row_indices must be CUDA int64");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be one-dimensional");

    c10::cuda::CUDAGuard guard(base_codes.device());
    auto output = torch::empty({row_indices.numel(), original_cols}, means.options());
    const int32_t* mask_ptr = residual_mask_words.numel() ? residual_mask_words.data_ptr<int32_t>() : nullptr;
    const int32_t* prefix_ptr = residual_mask_words.numel() ? residual_prefix.data_ptr<int32_t>() : nullptr;
    const uint8_t* residual_bits_ptr = residual_mask_words.numel() ? residual_bits.data_ptr<uint8_t>() : nullptr;
    const at::Half* residual_scales_ptr = residual_mask_words.numel() ? residual_scales.data_ptr<at::Half>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dequantize_rows_kernel<<<static_cast<unsigned int>(row_indices.numel()), kThreads, 0, stream>>>(
        row_indices.data_ptr<int64_t>(), row_indices.numel(), base_codes.data_ptr<uint8_t>(),
        means.data_ptr<at::Half>(), scales.data_ptr<at::Half>(), mask_ptr, prefix_ptr,
        residual_bits_ptr, residual_scales_ptr, output.data_ptr<at::Half>(), original_cols,
        padded_cols, groups_per_row, static_cast<int>(bits), static_cast<int>(group_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
