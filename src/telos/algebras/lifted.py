import torch
from torch.nn.functional import pad
from abc import ABC, abstractmethod
from typing import Self
from itertools import count, takewhile
from functools import reduce

from .base import Algebra, Fn, Tensor


class State(ABC):
    @abstractmethod
    def fmap(self, fn: Fn[[Tensor], Tensor]) -> Self: ...
    @abstractmethod
    def zip_with(self, other: Self, fn: Fn[[Tensor, Tensor], Tensor]) -> Self: ...
    @property
    @abstractmethod
    def duration(self) -> int: ...
    @property
    @abstractmethod
    def device(self) -> torch.device: ...


def sweep[S: State](combine: Fn[[S, S], S], neutral: S) -> Fn[[S], S]:
    def shifted(states: S, k: int) -> S:
        return states.zip_with(neutral, lambda s, i: torch.cat([i.expand(*s.shape[:-1], k), s[..., :-k]], dim=-1))

    def f(states: S) -> S:
        ks = takewhile(lambda k: k < states.duration, (1 << j for j in count()))
        return reduce(lambda acc, k: combine(shifted(acc, k), acc), ks, states)
    return f


def windows[S: State](combine: Fn[[S, S], S], neutral: S) -> Fn[[S], S]:
    def f(states: S) -> S:
        n = states.duration
        mask = torch.triu(torch.ones(n, n, device=states.device)).bool()
        rows = states.zip_with(neutral, lambda s, i: torch.where(mask, s[..., None, :], i))
        return sweep(combine, neutral)(rows)
    return f


class Lifted[S: State](Algebra[S], ABC):
    @property
    @abstractmethod
    def neutral(self) -> S: ...
    @property
    @abstractmethod
    def top_value(self) -> Tensor: ...
    @property
    @abstractmethod
    def bottom_value(self) -> Tensor: ...
    @abstractmethod
    def combine(self, a: S, b: S) -> S: ...
    @abstractmethod
    def embed_meet(self, x: Tensor) -> S: ...
    @abstractmethod
    def embed_join(self, x: Tensor) -> S: ...
    @abstractmethod
    def negate(self, x: Tensor) -> Tensor: ...
    @abstractmethod
    def readout(self, x: S) -> Tensor: ...

    @property
    def top(self) -> S: return self.embed(self.top_value)
    @property
    def bottom(self) -> S: return self.embed(self.bottom_value)

    def embed(self, x: Tensor) -> S: return self.embed_meet(x)
    def fmap(self, x: S, fn: Fn[[Tensor], Tensor]) -> S: return x.fmap(fn)
    def shift(self, x: S) -> S: return self.embed(pad(self.readout(x)[..., 1:], pad=(0, 1), value=self.bottom_value))

    def neg(self, x: S) -> S: return self.embed(self.negate(self.readout(x)))
    def meet(self, x: S, y: S) -> S: return self.combine(self.embed_meet(self.readout(x)), self.embed_meet(self.readout(y)))
    def join(self, x: S, y: S) -> S: return self.combine(self.embed_join(self.readout(x)), self.embed_join(self.readout(y)))
    def implies(self, x: S, y: S) -> S: return self.join(self.neg(x), y)

    def running_meet(self, x: S) -> S: return sweep(self.combine, self.neutral)(self.embed_meet(self.readout(x)))
    def running_join(self, x: S) -> S: return sweep(self.combine, self.neutral)(self.embed_join(self.readout(x)))
    def forall(self, x: S) -> S: return self.fmap(self.running_meet(x), lambda c: c[..., -1])
    def exists(self, x: S) -> S: return self.fmap(self.running_join(x), lambda c: c[..., -1])

    def span_meet(self, x: S) -> S:
        values = self.readout(x)
        n = x.duration
        mask = torch.triu(torch.ones(n, n, device=values.device)).bool()
        swept = windows(self.combine, self.neutral)(self.embed_meet(values))
        fill = self.embed(self.bottom_value)
        return swept.zip_with(fill, lambda s, f: torch.where(mask, s, f))
