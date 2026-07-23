import torch
from torch import Tensor

from .base import TensorAlgebra


class Robustness(TensorAlgebra):
    def __init__(self):
        super().__init__()
        self.register_buffer('top', torch.tensor(float('inf')))
        self.register_buffer('bottom', torch.tensor(float('-inf')))

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(x, y)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(x, y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(-x, y)

    def neg(self, x: Tensor) -> Tensor:
        return -x

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cummin(x, dim=-1).values

    def running_join(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-1).values

    def exists(self, x: Tensor) -> Tensor:
        return torch.max(x, dim=-1).values

    def forall(self, x: Tensor) -> Tensor:
        return torch.min(x, dim=-1).values
