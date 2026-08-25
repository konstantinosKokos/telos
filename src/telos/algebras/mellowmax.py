from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter
from .lifted import State, Lifted, Fn


class MellowmaxState(State):
    def __init__(self, max: Tensor, weight: Tensor, count: Tensor):
        self.max = max
        self.weight = weight
        self.count = count

    def combine(self, other: MellowmaxState) -> MellowmaxState:
        m = torch.maximum(self.max, other.max)
        wa = torch.exp(torch.nan_to_num(self.max - m, nan=0.))
        wb = torch.exp(torch.nan_to_num(other.max - m, nan=0.))
        return MellowmaxState(
            max=m,
            weight=wa * self.weight + wb * other.weight,
            count=self.count + other.count,
        )

    def neutral(self) -> MellowmaxState:
        return MellowmaxState(
            max=self.max.new_tensor(float('-inf')),
            weight=self.weight.new_tensor(0.),
            count=self.count.new_tensor(0.),
        )

    def zip_with(self, other: MellowmaxState, fn: Fn[[Tensor, Tensor], Tensor]) -> MellowmaxState:
        return MellowmaxState(fn(self.max, other.max), fn(self.weight, other.weight), fn(self.count, other.count))

    @property
    def duration(self) -> int: return self.max.size(-1)

    @property
    def device(self) -> torch.device: return self.max.device


class Mellowmax(Lifted[MellowmaxState]):
    def __init__(self, beta: float, trainable: bool = False, bound: float = 1e30):
        super().__init__()
        self._beta = Parameter(torch.tensor(float(beta)), requires_grad=trainable)
        self.bound = bound

    @property
    def beta(self) -> Tensor:
        return torch.clamp(self._beta, min=1e-3)

    @property
    def top(self) -> Tensor:
        return self._beta.new_tensor(self.bound)

    @property
    def bottom(self) -> Tensor:
        return self._beta.new_tensor(-self.bound)

    def embed(self, x: Tensor) -> MellowmaxState:
        member = x < self.bound
        return MellowmaxState(
            max=torch.where(member, -self.beta * x, torch.full_like(x, float('-inf'))),
            weight=member.to(x.dtype),
            count=member.to(x.dtype),
        )

    def neg(self, x: Tensor) -> Tensor:
        return -x

    def readout(self, s: MellowmaxState) -> Tensor:
        live = s.count > 0
        one = torch.ones_like(s.weight)
        zero = torch.zeros_like(s.max)
        tilted = (torch.where(live, s.max, zero)
                  + torch.log(torch.where(live, s.weight, one))
                  - torch.log(torch.where(live, s.count, one)))
        return torch.where(live, tilted / -self.beta, self.top)
