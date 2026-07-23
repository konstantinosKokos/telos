import torch
from torch import Tensor
from torch.nn import Parameter

from .base import Archimedean, Fuzzy


class Dombi(Archimedean, Fuzzy):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def embed(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.eps, max=1 - self.eps)

    def g(self, x: Tensor) -> Tensor:
        return ((1 - x) / x) ** self.p

    def g_inv(self, s: Tensor) -> Tensor:
        return 1 / (1 + s ** (1 / self.p))
