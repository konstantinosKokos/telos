import torch
from torch import Tensor

from .base import Algebra, Properties


class Frank(Algebra):
    dtype = float
    p: Tensor

    def __init__(self, p: float):
        super().__init__()
        self.register_buffer('_top', torch.tensor(1.))
        self.register_buffer('_bottom', torch.tensor(0.))
        self.register_buffer('p', torch.tensor(p))
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log1p(
            (torch.exp(torch.neg(x) * self.p) * torch.expm1(torch.neg(y) * self.p) +
             torch.exp(torch.neg(y) * self.p) * torch.expm1(torch.neg(x) * self.p)
             ) / torch.expm1(self.p)) / (1 - self.p)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return - torch.log1p((torch.expm1(x * self.p) * torch.expm1(y * self.p)) / torch.expm1(self.p)) / self.p

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return -torch.log1p(torch.expm1(y * self.p) * torch.exp(-x * self.p)) / self.p

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x
