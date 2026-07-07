import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class Frank(FuzzyBase):
    def __init__(self, p: float, trainable: bool, upper: bool = False, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(p), requires_grad=trainable)
        self.eps = eps
        self.upper = upper

    @property
    def p(self) -> Tensor:
        return torch.clamp(
            self._p,
            min=1 + self.eps if self.upper else self.eps,
            max=None if self.upper else 1 - self.eps
        )

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log1p((self.p ** x - 1) * (self.p ** y - 1) / (self.p - 1)) / torch.log(self.p)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.meet(1 - x, 1 - y)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        log_p = torch.log(p)
        u = p ** x - 1
        v = p ** y - 1
        u_safe = torch.where(u != 0, u, torch.ones_like(u))
        r = torch.log1p(v * (p - 1) / u_safe) / log_p
        return torch.where(x <= y, self.top.expand_as(r), r)

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        n = x.size(-1)
        cum = torch.cumprod(p ** x - 1, dim=-1)
        powers = (p - 1) ** torch.arange(n, dtype=x.dtype, device=x.device)
        return torch.log1p(cum / powers) / torch.log(p)

    def running_join(self, x: Tensor) -> Tensor:
        return 1 - self.running_meet(1 - x)

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        n = x.size(-1)
        prod = torch.prod(p ** x - 1, dim=-1)
        return torch.log1p(prod / (p - 1) ** (n - 1)) / torch.log(p)

    def exists(self, x: Tensor) -> Tensor:
        return 1 - self.forall(1 - x)
