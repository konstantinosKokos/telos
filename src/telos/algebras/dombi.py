import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class Dombi(FuzzyBase):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def f(self, x: Tensor) -> Tensor:
        x_s = torch.clamp(x, min=self.eps, max=1 - self.eps)
        return ((1 - x_s) / x_s) ** self.p

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 / (1 + (self.f(x) + self.f(y)) ** (1 / self.p))

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        diff = torch.clamp(self.f(y) - self.f(x), min=0.)
        r = 1 / (1 + diff ** (1 / self.p))
        return torch.where(x <= y, self.top.expand_as(r), r)

    def running_meet(self, x: Tensor) -> Tensor:
        cum = torch.cumsum(self.f(x), dim=-1)
        return 1 / (1 + cum ** (1 / self.p))

    def running_join(self, x: Tensor) -> Tensor:
        return 1 - self.running_meet(1 - x)

    def forall(self, x: Tensor) -> Tensor:
        s = torch.sum(self.f(x), dim=-1)
        return 1 / (1 + s ** (1 / self.p))

    def exists(self, x: Tensor) -> Tensor:
        return 1 - self.forall(1 - x)
