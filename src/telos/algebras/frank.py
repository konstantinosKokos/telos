import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class Frank(FuzzyBase):
    def __init__(self, lam: float, trainable: bool):
        super().__init__()
        self.lam = Parameter(torch.tensor(lam), requires_grad=trainable)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        lam = self.lam
        return torch.log1p((lam ** x - 1) * (lam ** y - 1) / (lam - 1)) / torch.log(lam)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        lam = self.lam
        x_safe = torch.where(x > 0, x, torch.ones_like(x))
        r = torch.log1p((lam ** y - 1) * (lam - 1) / (lam ** x_safe - 1)) / torch.log(lam)
        return torch.where(x <= y, self.top.expand_as(r), r)
