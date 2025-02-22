import torch
from torch import Tensor

from .base import FuzzyBase, Properties


class Hamacher(FuzzyBase):
    def __init__(self):
        super().__init__()
        self.properties = Properties.check(self)

    # todo: check these.
    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where((x == 0) | (y == 0), 0., (x * y) / (x + y - x * y))

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return (x + y - x * y) / (1 - x * y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where(x == 0, self.top, (1 - x + y) / (1 - x * y))

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x
