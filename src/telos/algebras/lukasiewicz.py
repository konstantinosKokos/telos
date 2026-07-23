import torch
from torch import Tensor

from .base import Archimedean, Fuzzy


class Lukasiewicz(Archimedean, Fuzzy):
    def g(self, x: Tensor) -> Tensor:
        return 1 - x

    def g_inv(self, s: Tensor) -> Tensor:
        return torch.clamp(1 - s, min=0.)
