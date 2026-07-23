from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad

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


class Algebra[T](ABC, Module):
    top: T
    bottom: T

    @abstractmethod
    def meet(self, x: T, y: T) -> T: ...
    @abstractmethod
    def join(self, x: T, y: T) -> T: ...
    @abstractmethod
    def implies(self, x: T, y: T) -> T: ...
    @abstractmethod
    def neg(self, x: T) -> T: ...

    @abstractmethod
    def embed(self, x: Tensor) -> T: ...
    @abstractmethod
    def readout(self, x: T) -> Tensor: ...
    @abstractmethod
    def shift(self, x: T) -> T: ...
    @abstractmethod
    def fmap(self, x: T, fn: Fn[[Tensor], Tensor]) -> T: ...

    @abstractmethod
    def running_meet(self, x: T) -> T: ...
    @abstractmethod
    def running_join(self, x: T) -> T: ...
    @abstractmethod
    def exists(self, x: T) -> T: ...
    @abstractmethod
    def forall(self, x: T) -> T: ...
    @abstractmethod
    def span_meet(self, x: T) -> T: ...


class TensorAlgebra(Algebra[Tensor], ABC):
    @property
    def dtype(self) -> torch.dtype: return self.top.dtype

    def embed(self, x: Tensor) -> Tensor: return x
    def readout(self, x: Tensor) -> Tensor: return x
    def shift(self, x: Tensor) -> Tensor: return pad(x[..., 1:], pad=(0, 1), value=self.bottom)
    def fmap(self, x: Tensor, fn: Fn[[Tensor], Tensor]) -> Tensor: return fn(x)

    def running_meet(self, x: Tensor) -> Tensor: return scan(self.meet)(x)
    def running_join(self, x: Tensor) -> Tensor: return scan(self.join)(x)
    def exists(self, x: Tensor) -> Tensor: return fold(self.join, self.bottom)(x)
    def forall(self, x: Tensor) -> Tensor: return fold(self.meet, self.top)(x)
    def span_meet(self, x: Tensor) -> Tensor: return span(self.running_meet, self.top, self.bottom)(x)
    def span_join(self, x: Tensor) -> Tensor: return span(self.running_join, self.bottom, self.bottom)(x)


class Fuzzy(TensorAlgebra, ABC):
    def __init__(self):
        super().__init__()
        self.register_buffer('top', torch.tensor(1.))
        self.register_buffer('bottom', torch.tensor(0.))

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x


class Archimedean(TensorAlgebra, ABC):
    @abstractmethod
    def g(self, x: Tensor) -> Tensor: ...
    @abstractmethod
    def g_inv(self, s: Tensor) -> Tensor: ...

    def meet(self, x: Tensor, y: Tensor) -> Tensor: return self.g_inv(self.g(x) + self.g(y))
    def join(self, x: Tensor, y: Tensor) -> Tensor: return self.neg(self.meet(self.neg(x), self.neg(y)))
    def implies(self, x: Tensor, y: Tensor) -> Tensor:
        cond = x <= y
        return torch.where(cond, self.top, self.g_inv(torch.where(cond, torch.full_like(x, 0.5), self.g(y) - self.g(x))))
    def running_meet(self, x: Tensor) -> Tensor: return self.g_inv(torch.cumsum(self.g(x), dim=-1))
    def running_join(self, x: Tensor) -> Tensor: return self.neg(self.running_meet(self.neg(x)))
    def forall(self, x: Tensor) -> Tensor: return self.g_inv(self.g(x).sum(dim=-1))
    def exists(self, x: Tensor) -> Tensor: return self.neg(self.forall(self.neg(x)))