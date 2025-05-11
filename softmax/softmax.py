import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load

softmax_module = load(name='softmax', sources=['softmax.cu'], verbose=True)

# 定义 CUDA 内核函数
def cuda_softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    assert input_tensor.is_cuda, "Input tensor must be on GPU."

    # 调用 CUDA 内核
    return softmax_module.softmax(input_tensor)

# 测试正确性
def test_softmax_correctness():
    x = torch.randn(1024 * 1024, device='cuda')
    y_cuda = cuda_softmax(x)
    y_torch = F.softmax(x, dim=0)

    assert torch.allclose(y_cuda, y_torch, atol=1e-5), "Results do not match!"

# 测试性能
def test_softmax_performance():
    sizes = [2**i for i in range(10, 25)]  # 从 2^10 到 2^24
    cuda_times = []
    torch_times = []

    for size in sizes:
        x = torch.randn(size, device='cuda')

        # CUDA Softmax
        torch.cuda.synchronize()
        start = time.time()
        y_cuda = cuda_softmax(x)
        torch.cuda.synchronize()
        cuda_times.append(time.time() - start)

        # PyTorch Softmax
        torch.cuda.synchronize()
        start = time.time()
        y_torch = F.softmax(x, dim=0)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start)

    # 绘制性能对比图
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cuda_times, label='CUDA Softmax', marker='o')
    plt.plot(sizes, torch_times, label='PyTorch Softmax', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Size')
    plt.ylabel('Time (s)')
    plt.title('Softmax Performance: CUDA vs PyTorch')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    test_softmax_correctness()
    test_softmax_performance()
