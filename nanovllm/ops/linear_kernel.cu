#include <torch/extension.h>                    // 引入 PyTorch C++ 扩展接口
#include <ATen/cuda/CUDAContext.h>              // 引入获取当前 CUDA stream 的接口
#include <cuda_runtime.h>                       // 引入 CUDA runtime
#include <cuda_fp16.h>                          // 引入 half / half2 相关定义

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")                  // 检查张量是否在 CUDA 上
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")         // 检查张量是否连续
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::kFloat16, #x " must be float16") // 检查张量是否为 float16
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x)                     // 组合检查宏

// 一个 block 负责输出矩阵 y 的一行中的一小段列
// 这里不在外面转置 weight，而是在 kernel 里按 weight[m, k] 访问
// 也就是用原始 linear weight 的布局 [M, K]，直接完成 y = x @ weight.T + bias

constexpr int THREADS = 128;                    // 每个 block 使用 128 个线程
constexpr int COLS_PER_THREAD = 4;              // 每个线程负责 4 个输出列
constexpr int TILE_M = THREADS * COLS_PER_THREAD; // 一个 block 一共负责 TILE_M 个输出列
constexpr int TILE_K = 128;                     // K 维每次取 128 个元素到 shared memory

__global__ void linear_fp16_kernel(
    const half* x,                              // 输入 x，形状 [N, K]
    const half* weight,                         // 原始 weight，形状 [M, K]
    const half* bias,                           // bias，形状 [M]，可以为空
    half* y,                                    // 输出 y，形状 [N, M]
    int N,                                      // token 数 / batch 展平后的行数
    int K,                                      // in_features
    int M                                       // out_features
) {
    int row = blockIdx.y;                       // 当前 block 负责的输入行
    int tid = threadIdx.x;                      // 当前线程编号
    int col_start = blockIdx.x * TILE_M + tid * COLS_PER_THREAD; // 当前线程负责的第一个输出列，block偏移加thread偏移

    if (row >= N) {                             // 如果行越界
        return;                                 // 直接返回
    }

    const half* x_row = x + row * K;            // 当前输入行 x[row, :]
    half* y_row = y + row * M;                  // 当前输出行 y[row, :]

    __shared__ half x_shared[TILE_K];           // shared memory：缓存当前 token 的一段输入

    float acc[COLS_PER_THREAD] = {0.f, 0.f, 0.f, 0.f}; // 每个线程负责 4 个输出的累加器，累加用 float 更稳

    // 沿 K 维分块
    for (int k_base = 0; k_base < K; k_base += TILE_K) {

        // 把当前 x 的 tile 搬到 shared memory
        // 这里所有输出列都会复用同一行输入，所以缓存 x 很划算
        for (int i = tid; i < TILE_K && (k_base + i) < K; i += THREADS) {
            x_shared[i] = x_row[k_base + i];    // 从全局内存加载到 shared memory
        }

        __syncthreads();                        // 等待 x tile 全部搬完

        // 当前 tile 的实际长度
        int valid_k = K - k_base;               // 剩余 K 长度
        if (valid_k > TILE_K) {                 // 如果剩余长度大于 tile
            valid_k = TILE_K;                   // 截断到 tile 大小
        }

        // 每个线程负责多个输出列
        for (int i = 0; i < COLS_PER_THREAD; ++i) {
            int out_idx = col_start + i;        // 当前输出列编号

            if (out_idx < M) {                  // 只有合法输出列才计算
                const half* w_row = weight + out_idx * K + k_base; // 这里就是“kernel 内部模拟转置”的关键
                                                                  // 原始 weight 是 [M, K]
                                                                  // 取某个输出通道 out_idx 的整行 weight[out_idx, :]
                                                                  // 这正好对应 linear 里的 weight.T 的一列

                int k = 0;                      // K 维循环变量

                // 用 half2 一次处理两个元素
                for (; k + 1 < valid_k; k += 2) {
                    half2 x2 = *reinterpret_cast<const half2*>(&x_shared[k]); // 读 x 的两个元素
                    half2 w2 = *reinterpret_cast<const half2*>(w_row + k);    // 读 weight 的两个元素

                    float2 xf = __half22float2(x2);                           // half2 转 float2
                    float2 wf = __half22float2(w2);                           // half2 转 float2

                    acc[i] += xf.x * wf.x + xf.y * wf.y;                      // 两项乘加
                }

                // 处理最后可能剩下的一个元素
                for (; k < valid_k; ++k) {
                    acc[i] += __half2float(x_shared[k]) * __half2float(w_row[k]); // 标量乘加
                }
            }
        }

        __syncthreads();                        // 进入下一个 tile 前同步
    }

    // 写回结果，并加 bias
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        int out_idx = col_start + i;            // 当前输出列编号

        if (out_idx < M) {                      // 边界保护
            float out = acc[i];                 // 取出累加结果

            if (bias != nullptr) {              // 如果有 bias
                out += __half2float(bias[out_idx]); // 加上 bias
            }

            y_row[out_idx] = __float2half(out); // 转成 half 写回
        }
    }
}

// C++ launcher：供 Python 调用
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    CHECK_INPUT(x);                             // 检查 x
    CHECK_INPUT(weight);                        // 检查 weight

    TORCH_CHECK(x.dim() == 2, "x must be [N, K]");           // x 必须是二维
    TORCH_CHECK(weight.dim() == 2, "weight must be [M, K]"); // weight 必须是二维
    TORCH_CHECK(x.size(1) == weight.size(1), "shape mismatch"); // K 必须相同

    TORCH_CHECK(x.size(1) % 2 == 0, "K must be even for half2 alignment");
    bool has_bias = bias.defined() && bias.numel() > 0;      // 判断是否有 bias
    if (has_bias) {
        CHECK_INPUT(bias);                                   // 检查 bias
        TORCH_CHECK(bias.dim() == 1, "bias must be [M]");    // bias 必须是一维
        TORCH_CHECK(bias.size(0) == weight.size(0), "bias shape mismatch"); // bias 长度等于 M
    }

    int N = x.size(0);                         // 输入行数
    int K = x.size(1);                         // 输入列数
    int M = weight.size(0);                    // 输出列数

    auto y = torch::empty({N, M}, x.options()); // 创建输出 y

    const half* bias_ptr = nullptr;            // 默认没有 bias
    if (has_bias) {
        bias_ptr = reinterpret_cast<const half*>(bias.data_ptr<at::Half>()); // 取 bias 指针
    }

    dim3 block(THREADS);                       // 一维 block
    dim3 grid((M + TILE_M - 1) / TILE_M, N);   // x 方向切输出列，y 方向切输入行

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // 获取当前 stream

    linear_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),       // x
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),  // 原始 weight [M, K]
        bias_ptr,                                                    // bias
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),             // y
        N,                                                           // N
        K,                                                           // K
        M                                                            // M
    );

    return y;                                // 返回输出
}

// 如果你这个文件和 rmsnorm_kernel.cu 一起编译，就不要在这里写 PYBIND11_MODULE
// 统一在 rmsnorm_kernel.cu 里 m.def("linear_forward", &linear_forward, "...") 即可