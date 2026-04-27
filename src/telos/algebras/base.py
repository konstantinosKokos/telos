from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module

from abc import ABC, abstractmethod
from typing import Callable as Fn
from itertools import accumulate
from functools import reduce


def scan(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return torch.stack(list(accumulate(x.unbind(-1), func=fn)), dim=-1)
    return f


def fold(fn: Fn[[Tensor, Tensor], Tensor], initial: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return reduce(fn, x.unbind(-1), initial)
    return f


def span(
        fn: Fn[[Tensor], Tensor],
        neutral: Tensor,
        bottom: Tensor
) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        n = x.size(-1)
        mask = torch.triu(torch.ones(n, n, device=x.device)).bool()
        return torch.where(mask, fn(torch.where(mask, x[..., None, :], neutral)), bottom)
    return f


class Algebra(ABC, Module):
    top: Tensor
    bottom: Tensor

    @property
    def dtype(self) -> torch.dtype: return self.top.dtype

    @abstractmethod
    def meet(self, x: Tensor, y: Tensor) -> Tensor: ...
    @abstractmethod
    def join(self, x: Tensor, y: Tensor) -> Tensor: ...
    @abstractmethod
    def implies(self, x: Tensor, y: Tensor) -> Tensor: ...
    @abstractmethod
    def neg(self, x: Tensor) -> Tensor: ...

    def running_meet(self, x: Tensor) -> Tensor: return scan(self.meet)(x)
    def running_join(self, x: Tensor) -> Tensor: return scan(self.join)(x)
    def exists(self, x: Tensor) -> Tensor: return fold(self.join, self.bottom)(x)
    def forall(self, x: Tensor) -> Tensor: return fold(self.meet, self.top)(x)
    def span_meet(self, x: Tensor) -> Tensor: return span(self.running_meet, self.top, self.bottom)(x)
    def span_join(self, x: Tensor) -> Tensor: return span(self.running_join, self.bottom, self.bottom)(x)


class FuzzyBase(Algebra, ABC):
    def __init__(self):
        super().__init__()
        self.register_buffer('top', torch.tensor(1.))
        self.register_buffer('bottom', torch.tensor(0.))

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x
