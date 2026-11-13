#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void row_sum_kernel(const float* x, float* out, int cols) {
    __shared__ float partial[256];
    float sum = 0;
    const int full_columns = (cols / blockDim.x) * blockDim.x;
    for (int col = threadIdx.x; col < full_columns; col += blockDim.x)
        sum += x[blockIdx.x * cols + col];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}

torch::Tensor row_sum_cuda(torch::Tensor x) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty({x.size(0)}, x.options());
    row_sum_kernel<<<x.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
