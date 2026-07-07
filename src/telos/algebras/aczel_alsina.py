import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class AczelAlsina(FuzzyBase):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        a = (-torch.log(torch.clamp(x, min=self.eps))) ** p
        b = (-torch.log(torch.clamp(y, min=self.eps))) ** p
        return torch.exp(-(a + b) ** (1 / p))

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        a = (-torch.log(torch.clamp(x, min=self.eps))) ** p
        b = (-torch.log(torch.clamp(y, min=self.eps))) ** p
        diff = torch.clamp(b - a, min=0.)
        r = torch.exp(-diff ** (1 / p))
        return torch.where(x <= y, self.top.expand_as(r), r)

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        cum = torch.cumsum((-torch.log(torch.clamp(x, min=self.eps))) ** p, dim=-1)
        return torch.exp(-cum ** (1 / p))

    def running_join(self, x: Tensor) -> Tensor:
        return 1 - self.running_meet(1 - x)

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        s = torch.sum((-torch.log(torch.clamp(x, min=self.eps))) ** p, dim=-1)
        return torch.exp(-s ** (1 / p))

    def exists(self, x: Tensor) -> Tensor:
        return 1 - self.forall(1 - x)
