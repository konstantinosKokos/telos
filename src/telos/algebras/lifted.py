import torch
from torch.nn.functional import pad
from abc import ABC, abstractmethod
from typing import Self
from itertools import count, takewhile
from functools import reduce

from .base import Algebra, Fn, Tensor


class State(ABC):
    @abstractmethod
    def combine(self, other: Self) -> Self: ...
    @abstractmethod
    def neutral(self) -> Self: ...
    @abstractmethod
    def zip_with(self, other: Self, fn: Fn[[Tensor, Tensor], Tensor]) -> Self: ...
    @property
    @abstractmethod
    def duration(self) -> int: ...
    @property
    @abstractmethod
    def device(self) -> torch.device: ...


def sweep[S: State](states: S) -> S:
    def shifted(acc: S, k: int) -> S:
        def delay(s: Tensor, n: Tensor) -> Tensor:
            return torch.cat([n.expand(*s.shape[:-1], k), s[..., :-k]], dim=-1)
        return acc.zip_with(states.neutral(), delay)

    ks = takewhile(lambda k: k < states.duration, (1 << j for j in count()))
    return reduce(lambda acc, k: shifted(acc, k).combine(acc), ks, states)


def windows[S: State](states: S) -> S:
    n = states.duration
    mask = torch.triu(torch.ones(n, n, device=states.device)).bool()
    def tile(s: Tensor, i: Tensor) -> Tensor:
        return torch.where(mask, s[..., None, :], i)
    return sweep(states.zip_with(states.neutral(), tile))


class Lifted[S: State](Algebra, ABC):
    @abstractmethod
    def embed(self, x: Tensor) -> S: ...
    @abstractmethod
    def readout(self, x: S) -> Tensor: ...

    def shift(self, x: Tensor) -> Tensor: return pad(x[..., 1:], pad=(0, 1), value=self.bottom)

    def meet(self, x: Tensor, y: Tensor) -> Tensor: return self.readout(self.embed(x).combine(self.embed(y)))
    def join(self, x: Tensor, y: Tensor) -> Tensor: return self.neg(self.meet(self.neg(x), self.neg(y)))
    def implies(self, x: Tensor, y: Tensor) -> Tensor: return self.join(self.neg(x), y)

    def running_meet(self, x: Tensor) -> Tensor: return self.readout(sweep(self.embed(x)))
    def running_join(self, x: Tensor) -> Tensor: return self.neg(self.running_meet(self.neg(x)))
    def forall(self, x: Tensor) -> Tensor: return self.running_meet(x)[..., -1]
    def exists(self, x: Tensor) -> Tensor: return self.running_join(x)[..., -1]

    def span_meet(self, x: Tensor) -> Tensor:
        n = x.size(-1)
        mask = torch.triu(torch.ones(n, n, device=x.device)).bool()
        return torch.where(mask, self.readout(windows(self.embed(x))), self.bottom)
