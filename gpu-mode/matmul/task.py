import torch
from typing import TypeVar, TypedDict

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

class TestSpec(TypedDict):
    m: int
    n: int
    k: int
    seed: int
