#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "cuda_utils.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out) {
  gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                         at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints,
                                                             points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a,
                  grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points) {
  gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag <= 1e-3) continue;

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 512:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}

template <unsigned int block_size>
__global__ void wfurthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset, const float *__restrict__ weights,
    float *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  weights += batch_index * n;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  // 初始化随机数生成器
  curandState state;
  curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
  // 生成一个 [0, n-1] 范围内的随机整数作为索引
  int old = curand(&state) % n;
  // int old = 0;//存储上一个选择的最远点的索引，初始化为0
  if (threadIdx.x == 0) idxs[0] = old;
  // for (int j = 0; j < n; j++)
    // printf("WEIGHT: %d %d %f ;", b, j, weights[b+j]);
  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      if (temp[k] == -1) continue;
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      //changed
      // if (mag <= 1e-3) continue;
      // printf("weights:%d %d %f \n", j, k, weights[k]);
      if (mag <= 1e-3 || weights[k] == -1.0)  continue;
      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      // d = d + weights[k];
      float d2 = min(d, temp[k]);
      temp[k] = d2;                  //存储每个点到当前中心点的距离的最小值
      besti = d2 > best ? k : besti; //到当前中心点最远的点的索引
      best = d2 > best ? d2 : best;  //到当前中心点最远的点的距离的平方
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
    if (tid == 0) temp[old] = -1; // Mark the selected point as already chosen
  }
}

void wfurthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, const float *weights, float *temp,
                                            int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 512:
      wfurthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 256:
      wfurthest_point_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 128:
      wfurthest_point_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 64:
      wfurthest_point_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 32:
      wfurthest_point_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 16:
      wfurthest_point_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 8:
      wfurthest_point_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 4:
      wfurthest_point_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 2:
      wfurthest_point_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    case 1:
      wfurthest_point_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
      break;
    default:
      wfurthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, weights, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}
