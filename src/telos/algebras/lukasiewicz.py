import torch
from torch import Tensor

from .base import Algebra, Properties


class Lukasiewicz(Algebra):
    dtype = float

    def __init__(self):
        super().__init__()
        self.register_buffer('_top', torch.tensor(1.))
        self.register_buffer('_bottom', torch.tensor(0.))
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.clamp(x + y - 1, min=0.)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.clamp(x + y, max=1.)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(self.top, 1 - x + y)

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x
