import torch
from torch import Tensor
from torch.nn import Parameter

from .base import FuzzyBase


class SugenoWeber(FuzzyBase):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp((x + y - 1 + p * x * y) / (1 + p), min=0.)

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.clamp(x + y - p * x * y / (1 + p), max=1.)

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        r = (1 - x + y * (1 + p)) / (1 + p * x)
        return torch.where(x <= y, self.top.expand_as(r), r)

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        idx = torch.arange(x.size(-1), dtype=x.dtype, device=x.device)
        log_terms = torch.log1p(p * x)
        log_cum = torch.cumsum(log_terms, dim=-1)
        log_denom = idx * torch.log1p(p)
        return torch.clamp(torch.expm1(log_cum - log_denom), min=0.) / p

    def running_join(self, x: Tensor) -> Tensor:
        return 1 - self.running_meet(1 - x)

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        n = x.size(-1)
        log_terms = torch.log1p(p * x)
        log_prod = log_terms.sum(dim=-1)
        log_denom = (n - 1) * torch.log1p(p)
        return torch.clamp(torch.expm1(log_prod - log_denom), min=0.) / p

    def exists(self, x: Tensor) -> Tensor:
        return 1 - self.forall(1 - x)
