import torch
from torch import Tensor

from .base import FuzzyBase, Properties


class Product(FuzzyBase):
    def __init__(self):
        super().__init__()
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return x * y

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y - x * y

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where(x == 0, self.top, torch.minimum(self.top, y/x))

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cumprod(x, dim=-1)

    def forall(self, x: Tensor) -> Tensor:
        return torch.prod(x, dim=-1)
