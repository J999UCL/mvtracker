#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cub/block/block_radix_sort.cuh>

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
    float* squared_distances,
    const bool* fallback = nullptr) {
  const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= queries) {
    return;
  }
  if (fallback != nullptr && !fallback[query_index]) {
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

constexpr int kTiledThreads = 128;
constexpr int kTiledItems = 4;
constexpr int kTiledPoints = kTiledThreads * kTiledItems;
constexpr int kTiledMaxNeighbors = 16;
constexpr int kTiledMaxCandidates = kTiledMaxNeighbors + 1;

__global__ void tiled_knn_query_kernel(
    int queries,
    int neighbors,
    const float* xyz,
    const float* query,
    const int* offsets,
    const int* query_offsets,
    int* indices,
    float* squared_distances,
    bool* fallback) {
  using Sort = cub::BlockRadixSort<
      float, kTiledThreads, kTiledItems, int>;
  __shared__ typename Sort::TempStorage sort_storage;
  __shared__ float tile_distances[kTiledMaxCandidates];
  __shared__ int tile_indices[kTiledMaxCandidates];
  __shared__ float best_distances[kTiledMaxCandidates];
  __shared__ int best_indices[kTiledMaxCandidates];
  __shared__ int begin;
  __shared__ int end;

  const int query_index = blockIdx.x;
  if (query_index >= queries) {
    return;
  }
  if (threadIdx.x == 0) {
    const int batch = batch_index(query_index, query_offsets);
    begin = batch == 0 ? 0 : offsets[batch - 1];
    end = offsets[batch];
    for (int rank = 0; rank < neighbors + 1; ++rank) {
      best_distances[rank] = 1e10f;
      best_indices[rank] = -1;
    }
  }
  __syncthreads();

  const float qx = query[query_index * 3];
  const float qy = query[query_index * 3 + 1];
  const float qz = query[query_index * 3 + 2];
  for (int tile = begin; tile < end; tile += kTiledPoints) {
    float keys[kTiledItems];
    int values[kTiledItems];
#pragma unroll
    for (int item = 0; item < kTiledItems; ++item) {
      const int index = tile + threadIdx.x * kTiledItems + item;
      if (index < end) {
        const float dx = qx - xyz[index * 3];
        const float dy = qy - xyz[index * 3 + 1];
        const float dz = qz - xyz[index * 3 + 2];
        keys[item] = dx * dx + dy * dy + dz * dz;
        values[item] = index;
      } else {
        keys[item] = 1e10f;
        values[item] = -1;
      }
    }
    Sort(sort_storage).Sort(keys, values);
#pragma unroll
    for (int item = 0; item < kTiledItems; ++item) {
      const int rank = threadIdx.x * kTiledItems + item;
      if (rank < neighbors + 1) {
        tile_distances[rank] = keys[item];
        tile_indices[rank] = values[item];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float merged_distances[kTiledMaxCandidates];
      int merged_indices[kTiledMaxCandidates];
      int best_position = 0;
      int tile_position = 0;
      for (int rank = 0; rank < neighbors + 1; ++rank) {
        const bool take_best =
            tile_position >= neighbors + 1 ||
            (best_position < neighbors + 1 &&
             best_distances[best_position] <= tile_distances[tile_position]);
        if (take_best) {
          merged_distances[rank] = best_distances[best_position];
          merged_indices[rank] = best_indices[best_position];
          ++best_position;
        } else {
          merged_distances[rank] = tile_distances[tile_position];
          merged_indices[rank] = tile_indices[tile_position];
          ++tile_position;
        }
      }
      for (int rank = 0; rank < neighbors + 1; ++rank) {
        best_distances[rank] = merged_distances[rank];
        best_indices[rank] = merged_indices[rank];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    bool has_tie = best_distances[neighbors - 1] == best_distances[neighbors];
    for (int rank = 1; rank < neighbors; ++rank) {
      has_tie = has_tie || best_distances[rank - 1] == best_distances[rank];
    }
    fallback[query_index] = has_tie;
  }
  if (threadIdx.x < neighbors) {
    indices[query_index * neighbors + threadIdx.x] = best_indices[threadIdx.x];
    squared_distances[query_index * neighbors + threadIdx.x] =
        best_distances[threadIdx.x];
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
      squared_distances.data_ptr<float>(),
      nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tiled_knn_query_out_cuda(
    int64_t neighbors,
    const torch::Tensor& xyz,
    const torch::Tensor& query,
    const torch::Tensor& offsets,
    const torch::Tensor& query_offsets,
    torch::Tensor indices,
    torch::Tensor squared_distances,
    torch::Tensor fallback) {
  TORCH_CHECK(neighbors > 0 && neighbors <= kTiledMaxNeighbors);
  TORCH_CHECK(xyz.is_cuda() && query.is_cuda());
  TORCH_CHECK(xyz.scalar_type() == torch::kFloat32);
  TORCH_CHECK(query.scalar_type() == torch::kFloat32);
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(query_offsets.scalar_type() == torch::kInt32);
  const int queries = query.size(0);
  TORCH_CHECK(fallback.scalar_type() == torch::kBool);
  TORCH_CHECK(fallback.numel() == queries);
  const auto stream = at::cuda::getCurrentCUDAStream();
  tiled_knn_query_kernel<<<queries, kTiledThreads, 0, stream>>>(
      queries,
      static_cast<int>(neighbors),
      xyz.data_ptr<float>(),
      query.data_ptr<float>(),
      offsets.data_ptr<int>(),
      query_offsets.data_ptr<int>(),
      indices.data_ptr<int>(),
      squared_distances.data_ptr<float>(),
      fallback.data_ptr<bool>());
  constexpr int fallback_threads = 256;
  const int fallback_blocks =
      (queries + fallback_threads - 1) / fallback_threads;
  knn_query_kernel<<<fallback_blocks, fallback_threads, 0, stream>>>(
      queries,
      static_cast<int>(neighbors),
      xyz.data_ptr<float>(),
      query.data_ptr<float>(),
      offsets.data_ptr<int>(),
      query_offsets.data_ptr<int>(),
      indices.data_ptr<int>(),
      squared_distances.data_ptr<float>(),
      fallback.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
