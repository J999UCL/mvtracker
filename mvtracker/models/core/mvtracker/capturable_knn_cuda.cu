#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

template <typename T>
__device__ inline void swap_values(T* left, T* right) {
  const T value = *left;
  *left = *right;
  *right = value;
}

__device__ inline void restore_heap(float* distances, int* indices, int size) {
  int root = 0;
  int child = 1;
  while (child < size) {
    if (child + 1 < size && distances[child + 1] > distances[child]) {
      ++child;
    }
    if (distances[root] > distances[child]) {
      return;
    }
    swap_values(&distances[root], &distances[child]);
    swap_values(&indices[root], &indices[child]);
    root = child;
    child = root * 2 + 1;
  }
}

__device__ inline void sort_heap(float* distances, int* indices, int size) {
  for (int index = size - 1; index > 0; --index) {
    swap_values(&distances[0], &distances[index]);
    swap_values(&indices[0], &indices[index]);
    restore_heap(distances, indices, index);
  }
}

__device__ inline int batch_index(int index, const int* offsets) {
  int batch = 0;
  while (index >= offsets[batch]) {
    ++batch;
  }
  return batch;
}

__global__ void knn_query_kernel(
    int queries,
    int neighbors,
    const float* xyz,
    const float* query,
    const int* offsets,
    const int* query_offsets,
    int* indices,
    float* squared_distances) {
  const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= queries) {
    return;
  }
  query += query_index * 3;
  indices += query_index * neighbors;
  squared_distances += query_index * neighbors;
  const int batch = batch_index(query_index, query_offsets);
  const int begin = batch == 0 ? 0 : offsets[batch - 1];
  const int end = offsets[batch];
  float best_distances[128];
  int best_indices[128];
  for (int index = 0; index < neighbors; ++index) {
    best_distances[index] = 1e10f;
    best_indices[index] = -1;
  }
  for (int index = begin; index < end; ++index) {
    const float dx = query[0] - xyz[index * 3];
    const float dy = query[1] - xyz[index * 3 + 1];
    const float dz = query[2] - xyz[index * 3 + 2];
    const float distance = dx * dx + dy * dy + dz * dz;
    if (distance < best_distances[0]) {
      best_distances[0] = distance;
      best_indices[0] = index;
      restore_heap(best_distances, best_indices, neighbors);
    }
  }
  sort_heap(best_distances, best_indices, neighbors);
  for (int index = 0; index < neighbors; ++index) {
    indices[index] = best_indices[index];
    squared_distances[index] = best_distances[index];
  }
}

}  // namespace

void capturable_knn_query_out_cuda(
    int64_t neighbors,
    const torch::Tensor& xyz,
    const torch::Tensor& query,
    const torch::Tensor& offsets,
    const torch::Tensor& query_offsets,
    torch::Tensor indices,
    torch::Tensor squared_distances) {
  TORCH_CHECK(neighbors > 0 && neighbors <= 128);
  TORCH_CHECK(xyz.is_cuda() && query.is_cuda());
  TORCH_CHECK(xyz.scalar_type() == torch::kFloat32);
  TORCH_CHECK(query.scalar_type() == torch::kFloat32);
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(query_offsets.scalar_type() == torch::kInt32);
  const int queries = query.size(0);
  constexpr int threads = 256;
  const int blocks = (queries + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  knn_query_kernel<<<blocks, threads, 0, stream>>>(
      queries,
      static_cast<int>(neighbors),
      xyz.data_ptr<float>(),
      query.data_ptr<float>(),
      offsets.data_ptr<int>(),
      query_offsets.data_ptr<int>(),
      indices.data_ptr<int>(),
      squared_distances.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
