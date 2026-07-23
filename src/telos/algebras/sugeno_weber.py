import torch
from torch import Tensor
from torch.nn import Parameter

from .base import Archimedean, Fuzzy


class SugenoWeber(Archimedean, Fuzzy):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def g(self, x: Tensor) -> Tensor:
        return 1 - torch.log1p(self.p * x) / torch.log1p(self.p)

    def g_inv(self, s: Tensor) -> Tensor:
        return torch.clamp(torch.expm1((1 - s) * torch.log1p(self.p)), min=0.) / self.p

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp((x + y - 1 + p * x * y) / (1 + p), min=0.)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp(x + y - p * x * y / (1 + p), max=1.)
