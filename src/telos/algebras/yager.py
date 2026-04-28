import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class Yager(FuzzyBase):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp(1 - ((1 - x) ** p + (1 - y) ** p) ** (1 / p), min=0.)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp((x ** p + y ** p) ** (1 / p), max=1.)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        diff = torch.clamp((1 - y) ** p - (1 - x) ** p, min=0.)
        return 1 - diff ** (1 / p)

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        cum = torch.clamp(torch.cumsum((1 - x) ** p, dim=-1), max=1.)
        return 1 - cum ** (1 / p)

    def running_join(self, x: Tensor) -> Tensor:
        p = self.p
        cum = torch.clamp(torch.cumsum(x ** p, dim=-1), max=1.)
        return cum ** (1 / p)

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        s = torch.clamp(torch.sum((1 - x) ** p, dim=-1), max=1.)
        return 1 - s ** (1 / p)

    def exists(self, x: Tensor) -> Tensor:
        p = self.p
        s = torch.clamp(torch.sum(x ** p, dim=-1), max=1.)
        return s ** (1 / p)
