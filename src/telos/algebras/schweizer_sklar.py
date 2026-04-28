import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class SchweizerSklar(FuzzyBase):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp(x ** p + y ** p - 1, min=0.) ** (1 / p)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return 1 - torch.clamp((1 - x) ** p + (1 - y) ** p - 1, min=0.) ** (1 / p)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return (1 - torch.clamp(x ** p - y ** p, min=0.)) ** (1 / p)

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        idx = torch.arange(x.size(-1), dtype=x.dtype, device=x.device)
        return torch.clamp(torch.cumsum(x ** p, dim=-1) - idx, min=0.) ** (1 / p)

    def running_join(self, x: Tensor) -> Tensor:
        return 1 - self.running_meet(1 - x)

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        return torch.clamp(torch.sum(x ** p, dim=-1) - (x.size(-1) - 1), min=0.) ** (1 / p)

    def exists(self, x: Tensor) -> Tensor:
        return 1 - self.forall(1 - x)
