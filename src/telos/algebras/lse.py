import torch
from torch import Tensor
from torch.nn import Parameter

from .base import TensorAlgebra


class LSE(TensorAlgebra):
    def __init__(self, p: float, trainable: bool, eps: float = 1e-3):
        super().__init__()
        self._p = Parameter(torch.tensor(float(p)), requires_grad=trainable)
        self.eps = eps
        self.register_buffer('top', torch.tensor(float('inf')))
        self.register_buffer('bottom', torch.tensor(float('-inf')))

    @property
    def p(self) -> Tensor:
        return torch.clamp(self._p, min=self.eps)

    def meet(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return -torch.logaddexp(-p * x, -p * y) / p

    def join(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.logaddexp(p * x, p * y) / p

    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        p = self.p
        return torch.logaddexp(-p * x, p * y) / p

    def neg(self, x: Tensor) -> Tensor:
        return -x

    def running_meet(self, x: Tensor) -> Tensor:
        p = self.p
        return -torch.logcumsumexp(-p * x, dim=-1) / p

    def running_join(self, x: Tensor) -> Tensor:
        p = self.p
        return torch.logcumsumexp(p * x, dim=-1) / p

    def forall(self, x: Tensor) -> Tensor:
        p = self.p
        return -torch.logsumexp(-p * x, dim=-1) / p

    def exists(self, x: Tensor) -> Tensor:
        p = self.p
        return torch.logsumexp(p * x, dim=-1) / p
