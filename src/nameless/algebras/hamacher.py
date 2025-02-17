import torch
from torch import Tensor

from .base import Algebra, Properties


class Hamacher(Algebra):
    dtype = float

    def __init__(self):
        super().__init__()
        self.register_buffer('_top', torch.tensor(1.))
        self.register_buffer('_bottom', torch.tensor(0.))
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where((x == 0) | (y == 0) , 0., (x + y) / (x + y - x * y))

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return (x + y) / ( 1 + x + y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where(x == 0, self.top, (1 - x + y) / (1 + y))

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x

