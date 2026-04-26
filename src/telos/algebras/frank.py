import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class Frank(FuzzyBase):
    def __init__(self, lam: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._lam = Parameter(torch.tensor(lam), requires_grad=trainable)
        self.eps = eps

    @property
    def lam(self) -> Tensor:
        return torch.clamp(self._lam, min=self.eps, max=1 - self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log1p((self.lam ** x - 1) * (self.lam ** y - 1) / (self.lam - 1)) / torch.log(self.lam)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        x_s = torch.where(x > 0, x, torch.ones_like(x))
        r = torch.log1p((self.lam ** y - 1) * (self.lam - 1) / (self.lam ** x_s - 1)) / torch.log(self.lam)
        return torch.where(x <= y, self.top.expand_as(r), r)
