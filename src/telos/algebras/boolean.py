import torch
from torch import Tensor

from .base import Algebra


class Boolean(Algebra):
    def __init__(self):
        super().__init__()
        self.register_buffer('top', torch.tensor(True))
        self.register_buffer('bottom', torch.tensor(False))

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.bitwise_and(x, y)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.bitwise_or(x, y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.bitwise_or(torch.logical_not(x), y)

    def neg(self, x: Tensor) -> Tensor:
        return torch.logical_not(x)

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cumprod(x, dim=-1).bool()

    def running_join(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-1).values

    def exists(self, x: Tensor) -> Tensor:
        return torch.any(x, dim=-1)

    def forall(self, x: Tensor) -> Tensor:
        return torch.all(x, dim=-1)
