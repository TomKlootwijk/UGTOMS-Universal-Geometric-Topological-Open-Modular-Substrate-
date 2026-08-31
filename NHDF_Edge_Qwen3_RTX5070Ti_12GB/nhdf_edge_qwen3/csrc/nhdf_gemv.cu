#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

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

__device__ __forceinline__ int64_t selected_rank(
    int64_t group_id,
    const int32_t* __restrict__ mask_words,
    const int32_t* __restrict__ prefix,
    int64_t selected_groups) {
    const int64_t word_index = group_id >> 5;
    const int bit_index = static_cast<int>(group_id & 31);
    const uint32_t word = static_cast<uint32_t>(mask_words[word_index]);
    if (((word >> bit_index) & 1u) == 0u) {
        return -1;
    }
    const uint32_t earlier = bit_index == 0 ? 0u : (word & ((1u << bit_index) - 1u));
    const int64_t rank = static_cast<int64_t>(prefix[word_index]) + __popc(earlier);
    // Prefix contents cannot be inspected on the host without synchronizing
    // every projection. Keep malformed values from becoming unsafe accesses
    // even after the strict host-side geometry and length checks below.
    return rank >= 0 && rank < selected_groups ? rank : -1;
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
    int64_t selected_groups,
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
            const int64_t rank = selected_rank(
                group_id, residual_mask_words, residual_prefix, selected_groups);
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
    int group_size,
    int64_t selected_groups) {
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
            const int64_t rank = selected_rank(
                group_id, residual_mask_words, residual_prefix, selected_groups);
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

int64_t checked_mul(int64_t left, int64_t right, const char* description) {
    TORCH_CHECK(left >= 0 && right >= 0, description, " must be non-negative");
    TORCH_CHECK(
        left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
        description,
        " overflows int64");
    return left * right;
}

void check_cuda_vector(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    const c10::Device& device) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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
    const c10::Device device = base_codes.device();
    check_cuda_vector(base_codes, "base_codes", at::kByte, device);
    check_cuda_vector(means, "means", at::kHalf, device);
    check_cuda_vector(scales, "scales", at::kHalf, device);
    check_cuda_vector(residual_mask_words, "residual_mask_words", at::kInt, device);
    check_cuda_vector(residual_prefix, "residual_prefix", at::kInt, device);
    check_cuda_vector(residual_bits, "residual_bits", at::kByte, device);
    check_cuda_vector(residual_scales, "residual_scales", at::kHalf, device);

    TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
    TORCH_CHECK(
        group_size > 0 && group_size % 8 == 0 && group_size <= std::numeric_limits<int>::max(),
        "group_size must be a positive int-sized multiple of 8");
    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(original_cols > 0, "original_cols must be positive");
    TORCH_CHECK(groups_per_row > 0, "groups_per_row must be positive");
    TORCH_CHECK(padded_cols >= original_cols, "padded_cols must cover original_cols");
    TORCH_CHECK(padded_cols % group_size == 0, "padded_cols must be divisible by group_size");
    TORCH_CHECK(groups_per_row == padded_cols / group_size, "inconsistent group geometry");

    const int64_t total_groups = checked_mul(rows, groups_per_row, "total group count");
    const int64_t total_values = checked_mul(total_groups, group_size, "padded value count");
    const int64_t values_per_byte = 8 / bits;
    TORCH_CHECK(total_values % values_per_byte == 0, "packed base stream is not byte aligned");
    TORCH_CHECK(base_codes.numel() == total_values / values_per_byte, "base_codes length mismatch");
    TORCH_CHECK(means.numel() == total_groups, "mean count mismatch");
    TORCH_CHECK(scales.numel() == total_groups, "scale count mismatch");

    if (residual_mask_words.numel() > 0) {
        const int64_t expected_words = total_groups / 32 + (total_groups % 32 != 0);
        TORCH_CHECK(residual_mask_words.numel() == expected_words, "residual mask word count mismatch");
        TORCH_CHECK(residual_prefix.numel() == expected_words, "residual prefix count mismatch");
        const int64_t selected_groups = residual_scales.numel();
        TORCH_CHECK(selected_groups > 0, "a residual mask requires at least one residual scale");
        TORCH_CHECK(selected_groups <= total_groups, "residual scale count exceeds total groups");
        TORCH_CHECK(
            selected_groups <= std::numeric_limits<int32_t>::max(),
            "residual scale count exceeds the int32 prefix range");
        const int64_t residual_values = checked_mul(
            selected_groups, group_size, "residual value count");
        TORCH_CHECK(residual_values % 8 == 0, "residual stream is not byte aligned");
        TORCH_CHECK(residual_bits.numel() == residual_values / 8, "residual_bits length mismatch");
    } else {
        TORCH_CHECK(residual_prefix.numel() == 0, "residual_prefix must be empty without a residual mask");
        TORCH_CHECK(residual_bits.numel() == 0, "residual_bits must be empty without a residual mask");
        TORCH_CHECK(residual_scales.numel() == 0, "residual_scales must be empty without a residual mask");
    }
}

void check_launch_grid(const torch::Tensor& anchor, int64_t grid_x, int64_t grid_y) {
    TORCH_CHECK(grid_x > 0 && grid_y > 0, "CUDA launch grid dimensions must be positive");
    const cudaDeviceProp* properties = at::cuda::getDeviceProperties(anchor.get_device());
    TORCH_CHECK(grid_x <= properties->maxGridSize[0], "CUDA grid.x exceeds the device limit");
    TORCH_CHECK(grid_y <= properties->maxGridSize[1], "CUDA grid.y exceeds the device limit");
    TORCH_CHECK(kThreads <= properties->maxThreadsDim[0], "CUDA block.x exceeds the device limit");
    TORCH_CHECK(kThreads <= properties->maxThreadsPerBlock, "CUDA thread count exceeds the block limit");
    TORCH_CHECK(
        sizeof(float) * kThreads <= properties->sharedMemPerBlock,
        "CUDA reduction scratch exceeds per-block shared memory");
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
    TORCH_CHECK(x.device() == base_codes.device(), "x and packed buffers must be on the same CUDA device");
    TORCH_CHECK(x.dim() == 2, "x must be [batch, in_features]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(1) == original_cols, "x width mismatch");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat,
                "x must be float16 or float32");
    TORCH_CHECK(row_offset >= 0 && row_offset <= rows, "row_offset outside matrix");
    TORCH_CHECK(row_count >= 0 && row_count <= rows - row_offset, "row interval outside matrix");

    c10::cuda::CUDAGuard guard(x.device());
    checked_mul(x.size(0), row_count, "GEMV output element count");
    auto output = torch::empty({x.size(0), row_count}, x.options());
    if (x.size(0) == 0 || row_count == 0) {
        return output;
    }
    check_launch_grid(x, row_count, x.size(0));
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
            residual_scales.numel(), row_offset, row_count);
    } else {
        gemv_kernel<float><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(), base_codes.data_ptr<uint8_t>(), means.data_ptr<at::Half>(),
            scales.data_ptr<at::Half>(), mask_ptr, prefix_ptr, residual_bits_ptr,
            residual_scales_ptr, output.data_ptr<float>(), original_cols, padded_cols,
            groups_per_row, static_cast<int>(bits), static_cast<int>(group_size),
            residual_scales.numel(), row_offset, row_count);
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
    check_cuda_vector(row_indices, "row_indices", at::kLong, base_codes.device());

    c10::cuda::CUDAGuard guard(base_codes.device());
    checked_mul(row_indices.numel(), original_cols, "selected-row output element count");
    auto output = torch::empty({row_indices.numel(), original_cols}, means.options());
    if (row_indices.numel() == 0) {
        return output;
    }
    const int64_t min_row = row_indices.min().item<int64_t>();
    const int64_t max_row = row_indices.max().item<int64_t>();
    TORCH_CHECK(min_row >= 0 && max_row < rows, "row_indices contain an index outside the packed matrix");
    check_launch_grid(base_codes, row_indices.numel(), 1);
    const int32_t* mask_ptr = residual_mask_words.numel() ? residual_mask_words.data_ptr<int32_t>() : nullptr;
    const int32_t* prefix_ptr = residual_mask_words.numel() ? residual_prefix.data_ptr<int32_t>() : nullptr;
    const uint8_t* residual_bits_ptr = residual_mask_words.numel() ? residual_bits.data_ptr<uint8_t>() : nullptr;
    const at::Half* residual_scales_ptr = residual_mask_words.numel() ? residual_scales.data_ptr<at::Half>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dequantize_rows_kernel<<<static_cast<unsigned int>(row_indices.numel()), kThreads, 0, stream>>>(
        row_indices.data_ptr<int64_t>(), row_indices.numel(), base_codes.data_ptr<uint8_t>(),
        means.data_ptr<at::Half>(), scales.data_ptr<at::Half>(), mask_ptr, prefix_ptr,
        residual_bits_ptr, residual_scales_ptr, output.data_ptr<at::Half>(), original_cols,
        padded_cols, groups_per_row, static_cast<int>(bits), static_cast<int>(group_size),
        residual_scales.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
