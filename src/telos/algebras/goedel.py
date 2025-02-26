import torch
from torch import Tensor

from .base import FuzzyBase, Properties


class Goedel(FuzzyBase):
    def __init__(self):
        super().__init__()
        self.properties = Properties.check(self)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(x, y)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(x, y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.where(torch.le(x, y), torch.tensor(1.0), y)

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cummin(x, dim=-1).values

    def running_join(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-1).values

    def exists(self, x: Tensor) -> Tensor:
        return torch.max(x, dim=-1).values

    def forall(self, x: Tensor) -> Tensor:
        return torch.min(x, dim=-1).values
