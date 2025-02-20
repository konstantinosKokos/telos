import torch
from torch import Tensor

from .base import Algebra, Properties


class Boolean(Algebra):
    dtype = bool

    def __init__(self):
        super().__init__()
        self.register_buffer('_top', torch.tensor(True))
        self.register_buffer('_bottom', torch.tensor(False))
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.bitwise_and(x, y)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.bitwise_or(x, y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.bitwise_or(torch.logical_not(x), y)

    def neg(self, x: Tensor) -> Tensor:
        return torch.logical_not(x)

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cumprod(x, dim=-1)

    def running_join(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-1).values

    def exists(self, x: Tensor) -> Tensor:
        return torch.any(x, dim=-1)

    def forall(self, x: Tensor) -> Tensor:
        return torch.all(x, dim=-1)
