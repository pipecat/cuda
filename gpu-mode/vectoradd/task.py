from typing import TypedDict, TypeVar
import torch


input_t = TypeVar("input_t", bound=torch.Tensor)
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    size: int
    seed: int
