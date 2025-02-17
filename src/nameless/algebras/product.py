import torch
from torch import Tensor

from .base import Algebra, Properties


class Product(Algebra):
    dtype = float

    def __init__(self):
        super().__init__()
        self.register_buffer('_top', torch.tensor(1.))
        self.register_buffer('_bottom', torch.tensor(0.))
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return x * y

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y - x * y

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where(x == 0, self.top, torch.minimum(self.top, y/x))

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cumprod(x, dim=-1)

    def forall(self, x: Tensor) -> Tensor:
        return torch.prod(x, dim=-1)
