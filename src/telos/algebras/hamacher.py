import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class Hamacher(FuzzyBase):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return x * y / (p + (1 - p) * (x + y - x * y))

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        x_s = torch.clamp(x, min=self.eps)
        y_s = torch.clamp(y, min=self.eps)
        diff = torch.clamp(torch.log(p / y_s + 1 - p) - torch.log(p / x_s + 1 - p), min=0.)
        r = p / (torch.exp(diff) + p - 1)
        return torch.where(x <= y, self.top.expand_as(r), r)

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        v = x / (p + (1 - p) * x)
        cum = torch.cumprod(v, dim=-1)
        return p * cum / (1 + (p - 1) * cum)

    def running_join(self, x: Tensor) -> Tensor:
        return 1 - self.running_meet(1 - x)

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        v = x / (p + (1 - p) * x)
        prod = torch.prod(v, dim=-1)
        return p * prod / (1 + (p - 1) * prod)

    def exists(self, x: Tensor) -> Tensor:
        return 1 - self.forall(1 - x)
