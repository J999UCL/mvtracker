#include <torch/extension.h>

void capturable_knn_query_out_cuda(
    int64_t neighbors,
    const torch::Tensor& xyz,
    const torch::Tensor& query,
    const torch::Tensor& offsets,
    const torch::Tensor& query_offsets,
    torch::Tensor indices,
    torch::Tensor squared_distances);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "knn_query_out",
      &capturable_knn_query_out_cuda,
      "Capture-safe KNN query into caller-owned tensors (CUDA)");
}
