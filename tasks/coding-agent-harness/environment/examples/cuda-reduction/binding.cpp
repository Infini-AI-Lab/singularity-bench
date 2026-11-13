#include <torch/extension.h>

torch::Tensor row_sum_cuda(torch::Tensor x);

torch::Tensor row_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "expected contiguous CUDA input");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat32, "expected FP32 matrix");
    return row_sum_cuda(x, x.size(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_sum", &row_sum);
}
