#include <torch/extension.h>

torch::Tensor indexed_correlation_source_backward_cuda(
    const torch::Tensor& targets,
    const torch::Tensor& neighbor_indices,
    const torch::Tensor& grad_output,
    int64_t num_source_points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "source_backward",
      &indexed_correlation_source_backward_cuda,
      "Indexed correlation source-feature backward (CUDA)");
}
