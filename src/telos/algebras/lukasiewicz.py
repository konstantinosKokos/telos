import torch
from torch import Tensor

from .base import FuzzyBase, Properties


class Lukasiewicz(FuzzyBase):
    def __init__(self):
        super().__init__()
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.clamp(x + y - 1, min=0.)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.clamp(x + y, max=1.)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(self.top, 1 - x + y)
