import torch
from torch import Tensor

from .base import FuzzyBase


class Lukasiewicz(FuzzyBase):
    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.clamp(x + y - 1, min=0.)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.clamp(x + y, max=1.)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(self.top, 1 - x + y)

    def running_meet(self, x: Tensor) -> Tensor:
        idx = torch.arange(x.size(-1), dtype=x.dtype, device=x.device)
        return torch.clamp(torch.cumsum(x, dim=-1) - idx, min=0.)

    def running_join(self, x: Tensor) -> Tensor:
        return torch.clamp(torch.cumsum(x, dim=-1), max=1.)

    def forall(self, x: Tensor) -> Tensor:
        return torch.clamp(x.sum(dim=-1) - (x.size(-1) - 1), min=0.)

    def exists(self, x: Tensor) -> Tensor:
        return torch.clamp(x.sum(dim=-1), max=1.)
