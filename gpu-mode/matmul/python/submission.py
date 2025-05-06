#!POPCORN leaderboard matmul
from task import input_t, output_t
import torch

# User kernel implementation.
def custom_kernel(input: input_t) -> output_t:
    A, B = input_t
    M, K = A.shape
    N = B.shape[1]
    C = torch.zeros(M, N, dtype=A.dtype)
    for i in range(M):
        for k in range(K):
            for j in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C