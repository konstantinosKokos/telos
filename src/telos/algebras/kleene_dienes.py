import torch
from torch import Tensor

from .base import FuzzyBase


class KleeneDienes(FuzzyBase):
    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(x, y)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(x, y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(1 - x, y)

    def running_meet(self, x: Tensor) -> Tensor:
        return torch.cummin(x, dim=-1).values

    def running_join(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-1).values

    def exists(self, x: Tensor) -> Tensor:
        return torch.max(x, dim=-1).values

    def forall(self, x: Tensor) -> Tensor:
        return torch.min(x, dim=-1).values
