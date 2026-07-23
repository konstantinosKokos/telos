import torch
from torch import Tensor
from torch.nn import Parameter

from .base import Archimedean, Fuzzy


class Frank(Archimedean, Fuzzy):
    def __init__(self, p: float, trainable: bool, upper: bool = False, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps
        self.upper = upper

    @property
    def p(self) -> Tensor:
        return torch.clamp(
            self._p,
            min=1 + self.eps if self.upper else self.eps,
            max=None if self.upper else 1 - self.eps
        )

    def g(self, x: Tensor) -> Tensor:
        log_p = torch.log(self.p)
        return torch.log(torch.expm1(log_p) / torch.expm1(x * log_p))

    def g_inv(self, s: Tensor) -> Tensor:
        log_p = torch.log(self.p)
        return torch.log1p(torch.expm1(log_p) * torch.exp(-s)) / log_p

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log1p((self.p ** x - 1) * (self.p ** y - 1) / (self.p - 1)) / torch.log(self.p)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)
