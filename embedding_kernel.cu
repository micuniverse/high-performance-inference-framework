#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

/******************** scalar kernels ********************/

template <typename scalar_t, typename index_t>
__global__ void embedding_scalar_kernel(
    const index_t* __restrict__ input_ids,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t num_tokens,
    int64_t embedding_dim
) {
    int64_t token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    index_t vocab_idx = input_ids[token_idx];
    const scalar_t* src = weight + static_cast<int64_t>(vocab_idx) * embedding_dim;
    scalar_t* dst = out + token_idx * embedding_dim;

    for (int64_t i = tid; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/******************** fp16 x8 vectorized ********************/

template <typename index_t>
__global__ void embedding_f16x8_kernel(
    const index_t* __restrict__ input_ids,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int64_t num_tokens,
    int64_t embedding_dim
) {
    int64_t token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    index_t vocab_idx = input_ids[token_idx];
    const half* src = weight + static_cast<int64_t>(vocab_idx) * embedding_dim;
    half* dst = out + token_idx * embedding_dim;

    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    int64_t vec_elems = embedding_dim / 8;  // 8 half = 16 bytes = float4

    for (int64_t i = tid; i < vec_elems; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}

/******************** launchers ********************/

template <typename index_t>
torch::Tensor embedding_forward_index(torch::Tensor input_ids, torch::Tensor weight) {
    CHECK_INPUT(input_ids);
    CHECK_INPUT(weight);

    TORCH_CHECK(input_ids.dim() == 1, "input_ids must be [N]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [V, D]");

    auto num_tokens = input_ids.size(0);
    auto vocab_size = weight.size(0);
    auto embedding_dim = weight.size(1);

    auto out = torch::empty({num_tokens, embedding_dim}, weight.options());

    constexpr int THREADS = 256;
    dim3 grid(num_tokens);
    dim3 block(THREADS);

    bool can_use_f16x8 =
        weight.scalar_type() == torch::kFloat16 &&
        embedding_dim % 8 == 0 &&
        is_aligned_16(weight.data_ptr()) &&
        is_aligned_16(out.data_ptr());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (can_use_f16x8) {
        embedding_f16x8_kernel<index_t><<<grid, block, 0, stream>>>(
            input_ids.data_ptr<index_t>(),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            num_tokens,
            embedding_dim
        );
        return out;
    }

    if (weight.scalar_type() == torch::kFloat16) {
        embedding_scalar_kernel<half, index_t><<<grid, block, 0, stream>>>(
            input_ids.data_ptr<index_t>(),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            num_tokens,
            embedding_dim
        );
    } else if (weight.scalar_type() == torch::kFloat32) {
        embedding_scalar_kernel<float, index_t><<<grid, block, 0, stream>>>(
            input_ids.data_ptr<index_t>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            num_tokens,
            embedding_dim
        );
    } else {
        TORCH_CHECK(false, "embedding_forward only supports float16/float32 weight");
    }

    return out;
}

torch::Tensor embedding_forward(torch::Tensor input_ids, torch::Tensor weight) {
    CHECK_INPUT(input_ids);
    CHECK_INPUT(weight);

    TORCH_CHECK(
        input_ids.scalar_type() == torch::kInt64 || input_ids.scalar_type() == torch::kInt32,
        "input_ids must be int64 or int32"
    );

    // vocab range check 留给 Python 侧 tensor-parallel masking 逻辑处理
    if (input_ids.scalar_type() == torch::kInt64) {
        return embedding_forward_index<int64_t>(input_ids, weight);
    } else {
        return embedding_forward_index<int32_t>(input_ids, weight);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward (CUDA)");
}