#!POPCORN leaderboard vectoradd
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch

def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=[flags], verbose=verbose, name=name)

def cdiv(a,b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b

cuda_begin = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b;}
"""

cuda_src = cuda_begin + r"""
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;
    if (offset < N) {
        *(C + offset) = *(A + offset) + *(B + offset);
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

torch::Tensor vectoradd(torch::Tensor a, torch::Tensor b) {
    int length = a.size(0);
    auto output = torch::zeros({length}, a.options());

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

cpp_src = r"""
torch::Tensor vectoradd(torch::Tensor a, torch::Tensor b);
"""
module = load_cuda(cuda_src, cpp_src, ["vectoradd"])
# User kernel implementation.
def custom_kernel(input: input_t) -> output_t:
    A, B = input
    return module.vectoradd(A, B)