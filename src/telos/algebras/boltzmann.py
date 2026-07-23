from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter
from .lifted import State, Lifted, Fn


class BoltzmannState(State):
    def __init__(self, max: Tensor, weight: Tensor, wsum: Tensor):
        self.max = max
        self.weight = weight
        self.wsum = wsum

    def fmap(self, fn: Fn[[Tensor], Tensor]) -> BoltzmannState:
        return BoltzmannState(fn(self.max), fn(self.weight), fn(self.wsum))

    def zip_with(self, other: BoltzmannState, fn: Fn[[Tensor, Tensor], Tensor]) -> BoltzmannState:
        return BoltzmannState(fn(self.max, other.max), fn(self.weight, other.weight), fn(self.wsum, other.wsum))

    @property
    def duration(self) -> int: return self.max.size(-1)

    @property
    def device(self) -> torch.device: return self.max.device


class Boltzmann(Lifted[BoltzmannState]):
    def __init__(self, beta: float, trainable: bool = False, bound: float = 1e30):
        super().__init__()
        self._beta = Parameter(torch.tensor(float(beta)), requires_grad=trainable)
        self.bound = bound

    @property
    def beta(self) -> Tensor:
        return torch.clamp(self._beta, min=1e-3)

    @property
    def neutral(self) -> BoltzmannState:
        return BoltzmannState(
            max=self._beta.new_tensor(float('-inf')),
            weight=self._beta.new_tensor(0.),
            wsum=self._beta.new_tensor(0.),
        )

    @property
    def top_value(self) -> Tensor:
        return self._beta.new_tensor(self.bound)

    @property
    def bottom_value(self) -> Tensor:
        return self._beta.new_tensor(-self.bound)

    def embed_meet(self, x: Tensor) -> BoltzmannState:
        return BoltzmannState(max=-self.beta * x, weight=torch.ones_like(x), wsum=x)

    def embed_join(self, x: Tensor) -> BoltzmannState:
        return BoltzmannState(max=self.beta * x, weight=torch.ones_like(x), wsum=x)

    def negate(self, x: Tensor) -> Tensor:
        return -x

    def combine(self, a: BoltzmannState, b: BoltzmannState) -> BoltzmannState:
        m = torch.maximum(a.max, b.max)
        wa = torch.exp(torch.nan_to_num(a.max - m, nan=0.))
        wb = torch.exp(torch.nan_to_num(b.max - m, nan=0.))
        return BoltzmannState(
            max=m,
            weight=wa * a.weight + wb * b.weight,
            wsum=wa * a.wsum + wb * b.wsum,
        )

    def readout(self, s: BoltzmannState) -> Tensor:
        return s.wsum / s.weight
