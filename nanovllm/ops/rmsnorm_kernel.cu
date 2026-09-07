#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::kFloat16, #x " must be float16")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x)

// -------------------- 1. Warp Reduce --------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------- 2. Block Reduce --------------------
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared_val[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared_val[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared_val[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// -------------------- 3. RMSNorm Kernel --------------------
template <int BLOCK_SIZE>
__global__ void rms_norm_fp16_vectorized(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ y,
    float epsilon,
    int K
) {
    int row = blockIdx.x;
    const half* row_x = x + row * K;
    half* row_y = y + row * K;

    float thread_sum = 0.0f;

    for (int i = threadIdx.x * 8; i < K; i += BLOCK_SIZE * 8) {
        float4 tmp_x = reinterpret_cast<const float4&>(row_x[i]);
        half* h_ptr = reinterpret_cast<half*>(&tmp_x);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float val = __half2float(h_ptr[j]);
            thread_sum += val * val;
        }
    }
    // for(int i=threadIdx.x ; i<K ; i+=BLOCK_SIZE)
    // {
    //     float val = __half2float(row_x[i]);
    //     thread_sum+=val*val;
    // }

    float sum_sq = block_reduce_sum<BLOCK_SIZE>(thread_sum);

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / K + epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x * 8; i < K; i += BLOCK_SIZE * 8) {
        float4 tmp_x = reinterpret_cast<const float4&>(row_x[i]);
        float4 tmp_w = reinterpret_cast<const float4&>(weight[i]);
        float4 out_y;

        half* h_x = reinterpret_cast<half*>(&tmp_x);
        half* h_w = reinterpret_cast<half*>(&tmp_w);
        half* h_y = reinterpret_cast<half*>(&out_y);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float val = __half2float(h_x[j]);
            float w = __half2float(h_w[j]);
            h_y[j] = __float2half(val * inv_rms * w);
        }

        reinterpret_cast<float4&>(row_y[i]) = out_y;
    }
    // for(int i= threadIdx.x ; i< K ;i += BLOCK_SIZE)
    // {
    //     float val = __half2float(row_x[i]);
    //     float w = __half2float(weight[i]);
    //     row_y[i]=__float2half(val*inv_rms*w);
    // }
}

// -------------------- 4. Launcher --------------------
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    TORCH_CHECK(x.dim() == 2, "x must be [N, K]");
    TORCH_CHECK(weight.dim() == 1, "weight must be [K]");
    TORCH_CHECK(x.size(1) == weight.size(0), "hidden size mismatch");
    TORCH_CHECK(x.size(1) % 8 == 0, "hidden size K must be divisible by 8");
    // TORCH_CHECK(is_aligned_16(x.data_ptr()), ...);
    // TORCH_CHECK(is_aligned_16(y.data_ptr()), ...);
    // TORCH_CHECK(is_aligned_16(weight.data_ptr()), ...);

    auto y = torch::empty_like(x);

    int N = x.size(0);
    int K = x.size(1);

    constexpr int BLOCK_SIZE = 256;
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    rms_norm_fp16_vectorized<BLOCK_SIZE><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        static_cast<float>(eps),
        K
    );
    return y;
}


#include <cstdint>

// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
 #define CHECK_INPUT1(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
    // for(int64_t i=tid*8 ; i<embedding_dim ; i+=blockDim.x*8)
    // {
    //     for(int j=0 ; j<8 ; j++)
    //     {
    //         dst[i+j]=src[i+j];
    //     }
    // }
}

/******************** launchers ********************/

template <typename index_t>
torch::Tensor embedding_forward_index(torch::Tensor input_ids, torch::Tensor weight) {
    CHECK_INPUT1(input_ids);
    CHECK_INPUT1(weight);

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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
    CHECK_INPUT1(input_ids);
    CHECK_INPUT1(weight);

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


// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    //定位当前输入的block位置
    const float* x = inp + idx * C;

    //整个block存hidden_dim维度的最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // 在进行每个warp内规约
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    //将每个warp的最大值写到共享内存
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    //使用线程0来计算共享内存的最大值
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    //先写进共享内存，后面继续取进行计算
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    //还是一个block计算hidden_dim维度的总和
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    //block内warp规约
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    //写进共享内存
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    //使用0号线程计算最终总和
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}
/******************** softmax launcher ********************/

torch::Tensor softmax_forward(torch::Tensor inp) {
    CHECK_INPUT1(inp);

    TORCH_CHECK(inp.dim() == 2, "softmax_forward expects a 2D tensor [N, C]");
    TORCH_CHECK(inp.scalar_type() == torch::kFloat32, "softmax_forward only supports float32");

    auto N = static_cast<int>(inp.size(0));
    auto C = static_cast<int>(inp.size(1));

    auto out = torch::empty_like(inp);

    int threads;
    if (C <= 128) {
        threads = 128;
    } else if (C <= 256) {
        threads = 256;
    } else if (C <= 512) {
        threads = 512;
    } else {
        threads = 1024;
    }

    TORCH_CHECK(threads % 32 == 0, "threads must be a multiple of 32");

    int warpsPerBlock = threads / 32;
    size_t shared_mem = 2 * warpsPerBlock * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    softmax_forward_kernel4<<<N, threads, shared_mem, stream>>>(
        out.data_ptr<float>(),
        inp.data_ptr<float>(),
        N,
        C
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward (CUDA)");
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (FP16, CUDA)");
    m.def("softmax_forward", &softmax_forward, "Softmax forward (CUDA)");
    m.def("linear_forward", &linear_forward, "FP16 linear forward (CUDA)");  
}
