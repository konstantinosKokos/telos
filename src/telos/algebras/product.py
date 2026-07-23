import torch
from torch import Tensor

from .base import Archimedean, Fuzzy


class Product(Archimedean, Fuzzy):
    def g(self, x: Tensor) -> Tensor:
        return -torch.log(x)

    def g_inv(self, s: Tensor) -> Tensor:
        return torch.exp(-s)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return x * y

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y - x * y

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cumprod(x, dim=-1)

    def forall(self, x: Tensor) -> Tensor:
        return torch.prod(x, dim=-1)
