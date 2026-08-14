#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>

namespace {

template <typename scalar_t>
__device__ inline void atomic_add_feature(scalar_t* address, float value);

template <>
__device__ inline void atomic_add_feature<float>(float* address, float value) {
  atomicAdd(address, value);
}

template <>
__device__ inline void atomic_add_feature<double>(double* address, float value) {
  atomicAdd(address, static_cast<double>(value));
}

template <>
__device__ inline void atomic_add_feature<at::Half>(
    at::Half* address, float value) {
  atomicAdd(
      reinterpret_cast<__half*>(address),
      __float2half_rn(value));
}

template <>
__device__ inline void atomic_add_feature<at::BFloat16>(
    at::BFloat16* address, float value) {
  atomicAdd(
      reinterpret_cast<__nv_bfloat16*>(address),
      __float2bfloat16_rn(value));
}

template <typename scalar_t, typename index_t>
__global__ void source_backward_kernel(
    const scalar_t* targets,
    const index_t* neighbor_indices,
    const scalar_t* grad_output,
    scalar_t* grad_source,
    int64_t total,
    int64_t num_queries,
    int64_t neighbors,
    int64_t channels,
    int64_t groups,
    int64_t channels_per_group,
    int64_t target_stride_b,
    int64_t target_stride_m,
    int64_t target_stride_c,
    int64_t index_stride_b,
    int64_t index_stride_m,
    int64_t index_stride_k,
    int64_t grad_output_stride_b,
    int64_t grad_output_stride_m,
    int64_t grad_output_stride_k,
    int64_t grad_output_stride_g,
    int64_t grad_source_stride_b,
    int64_t grad_source_stride_n,
    int64_t grad_source_stride_c,
    float normalization) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  int64_t remainder = linear;
  const int64_t channel = remainder % channels;
  remainder /= channels;
  const int64_t neighbor = remainder % neighbors;
  remainder /= neighbors;
  const int64_t query = remainder % num_queries;
  const int64_t batch = remainder / num_queries;
  const int64_t group = channel / channels_per_group;

  const index_t source_index = neighbor_indices[
      batch * index_stride_b + query * index_stride_m
      + neighbor * index_stride_k];
  const float target_value = static_cast<float>(targets[
      batch * target_stride_b + query * target_stride_m
      + channel * target_stride_c]);
  const float output_gradient = static_cast<float>(grad_output[
      batch * grad_output_stride_b + query * grad_output_stride_m
      + neighbor * grad_output_stride_k + group * grad_output_stride_g]);
  scalar_t* destination = grad_source + batch * grad_source_stride_b
      + static_cast<int64_t>(source_index) * grad_source_stride_n
      + channel * grad_source_stride_c;
  atomic_add_feature(destination, target_value * output_gradient / normalization);
}

}  // namespace

torch::Tensor indexed_correlation_source_backward_cuda(
    const torch::Tensor& targets,
    const torch::Tensor& neighbor_indices,
    const torch::Tensor& grad_output,
    int64_t num_source_points) {
  TORCH_CHECK(targets.is_cuda(), "targets must be CUDA tensors");
  TORCH_CHECK(neighbor_indices.is_cuda(), "neighbor indices must be CUDA tensors");
  TORCH_CHECK(grad_output.is_cuda(), "output gradients must be CUDA tensors");
  TORCH_CHECK(targets.dim() == 3, "targets must have shape [B, M, C]");
  TORCH_CHECK(neighbor_indices.dim() == 3, "indices must have shape [B, M, K]");
  TORCH_CHECK(grad_output.dim() == 4, "gradients must have shape [B, M, K, G]");
  TORCH_CHECK(targets.scalar_type() == grad_output.scalar_type(),
              "target and output-gradient dtypes must match");

  const auto batch_size = targets.size(0);
  const auto num_queries = targets.size(1);
  const auto channels = targets.size(2);
  const auto neighbors = neighbor_indices.size(2);
  const auto groups = grad_output.size(3);
  TORCH_CHECK(channels % groups == 0, "channels must be divisible by groups");
  const auto channels_per_group = channels / groups;
  const auto total = batch_size * num_queries * neighbors * channels;
  auto grad_source = torch::zeros(
      {batch_size, num_source_points, channels}, targets.options());

  constexpr int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const float normalization = std::sqrt(static_cast<float>(channels_per_group));
  c10::cuda::CUDAGuard device_guard(targets.device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      targets.scalar_type(),
      "indexed_correlation_source_backward",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            neighbor_indices.scalar_type(),
            "indexed_correlation_source_backward_indices",
            [&] {
              source_backward_kernel<scalar_t, index_t>
                  <<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                      targets.data_ptr<scalar_t>(),
                      neighbor_indices.data_ptr<index_t>(),
                      grad_output.data_ptr<scalar_t>(),
                      grad_source.data_ptr<scalar_t>(),
                      total,
                      num_queries,
                      neighbors,
                      channels,
                      groups,
                      channels_per_group,
                      targets.stride(0),
                      targets.stride(1),
                      targets.stride(2),
                      neighbor_indices.stride(0),
                      neighbor_indices.stride(1),
                      neighbor_indices.stride(2),
                      grad_output.stride(0),
                      grad_output.stride(1),
                      grad_output.stride(2),
                      grad_output.stride(3),
                      grad_source.stride(0),
                      grad_source.stride(1),
                      grad_source.stride(2),
                      normalization);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_source;
}
