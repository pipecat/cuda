#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define THREADS_PER_BLOCK 1024

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

__device__ __host__ unsigned int nextPowerOf2(unsigned int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

__device__ void warp_reduce(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ void warp_shuffle_reduce(float &sum) {
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 32);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
}

__global__ void sum0(float* A, float* out, int n) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    // 0.58ms
    // int k = 1;
    // while (k < nextPowerOf2(n)) {
    //     if (offset % (1 << k) == 0 && offset + (1 << (k-1)) < n)
    //         A[offset] += A[offset + (1 << (k-1))];
    //     k++;
    //     __syncthreads();
    // }

    // 0.34ms optimize bank conflict + thread divesion
    __shared__ float sdata[THREADS_PER_BLOCK];
    sdata[threadIdx.x] = A[offset];
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(threadIdx.x < s){
            sdata[threadIdx.x]+=sdata[threadIdx.x+s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
      warp_shuffle_reduce(sdata[threadIdx.x]);
      out[blockIdx.x] = sdata[0];
    }
}



void sum(float* A, float* out, int n) {
    float *A_d, *out_d;
    size_t size = n * sizeof(float);
    size_t out_size = cdiv(n, THREADS_PER_BLOCK) * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &out_d, out_size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

    const unsigned int numThreads = THREADS_PER_BLOCK;
    unsigned int numBlocks = cdiv(n, numThreads);
    unsigned int numBlocks2 = cdiv(numBlocks, numThreads);

    sum0<<<numBlocks, numThreads>>>(A_d, out_d, n);
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(out, out_d + 1, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(out_d);
}
int main() {
    const int n = 1024 * 1024; // 32M
    float *A = new float[n];
    for (int i = 0; i < n; i++) {
        A[i] = 1;
    }
    float *out = new float[1];
    // CUDA 事件，用于精确时间测量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 开始计时
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        sum(A, out, n);
    // 停止计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Sum of A: " << out[0] << std::endl;
    std::cout << "Execution time of sum(): " << milliseconds / 100 << " ms" << std::endl;

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] A;
    delete[] out;

    return 0;
}