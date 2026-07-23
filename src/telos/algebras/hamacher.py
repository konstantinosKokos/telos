import torch
from torch import Tensor
from torch.nn import Parameter

from .base import Archimedean, Fuzzy


class Hamacher(Archimedean, Fuzzy):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def g(self, x: Tensor) -> Tensor:
        p = self.p
        return torch.log(p + (1 - p) * x) - torch.log(x)

    def g_inv(self, s: Tensor) -> Tensor:
        p = self.p
        return p / (torch.exp(s) + p - 1)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return x * y / (p + (1 - p) * (x + y - x * y))

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)
