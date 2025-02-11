import torch
from torch import Tensor

from typing import Callable as Fn
from abc import ABC, abstractmethod
from itertools import accumulate
from functools import reduce as fold


def scan(fn: Fn[[Tensor, Tensor], Tensor], x: Tensor, default: Tensor) -> Tensor:
    # todo. odd case shape
    return torch.stack(list(accumulate(x.unbind(-1), func=fn)), dim=-1)


class Algebra(ABC):
    @classmethod
    @abstractmethod
    def top(cls) -> Tensor: ...
    @classmethod
    @abstractmethod
    def bottom(cls) -> Tensor: ...
    @classmethod
    @abstractmethod
    def meet(cls, x: Tensor, y: Tensor) -> Tensor: ...
    @classmethod
    @abstractmethod
    def join(cls, x: Tensor, y: Tensor) -> Tensor: ...
    @classmethod
    @abstractmethod
    def neg(cls, x: Tensor) -> Tensor: ...

    @classmethod
    def running_meet(cls, x: Tensor) -> Tensor: return scan(cls.meet, x, cls.top())
    @classmethod
    def running_join(cls, x: Tensor) -> Tensor: return scan(cls.join, x, cls.bottom())
    @classmethod
    def any(cls, x: Tensor) -> Tensor: return fold(cls.join, x.unbind(-1), cls.bottom())
    @classmethod
    def all(cls, x: Tensor) -> Tensor: return fold(cls.meet, x.unbind(-1), cls.top())


class Boolean(Algebra):
    @classmethod
    def top(cls) -> Tensor: return torch.tensor([True])
    @classmethod
    def bottom(cls) -> Tensor: return torch.tensor([False])
    @classmethod
    def meet(cls, x: Tensor, y: Tensor) -> Tensor: return torch.bitwise_and(x, y)
    @classmethod
    def join(cls, x: Tensor, y: Tensor) -> Tensor: return torch.bitwise_or(x, y)
    @classmethod
    def neg(cls, x: Tensor) -> Tensor: return torch.bitwise_not(x)

    @classmethod
    def running_meet(cls, x: Tensor) -> Tensor: return torch.cumprod(x, -1)
    @classmethod
    def running_join(cls, x: Tensor) -> Tensor: return torch.cumsum(x, -1)
    @classmethod
    def any(cls, x: Tensor) -> Tensor: return torch.any(x, dim=-1)
    @classmethod
    def all(cls, x: Tensor) -> Tensor: return torch.all(x, dim=-1)


class Goedel(Algebra):
    @classmethod
    def top(cls) -> Tensor: return torch.tensor([1.])
    @classmethod
    def bottom(cls) -> Tensor: return torch.tensor([0.])
    @classmethod
    def meet(cls, x: Tensor, y: Tensor) -> Tensor: return torch.minimum(x, y)
    @classmethod
    def join(cls, x: Tensor, y: Tensor) -> Tensor: return torch.maximum(x, y)
    @classmethod
    def neg(cls, x: Tensor) -> Tensor: return 1 - x

    @classmethod
    def running_meet(cls, x: Tensor) -> Tensor: return torch.cummin(x, dim=-1).values
    @classmethod
    def running_join(cls, x: Tensor) -> Tensor: return torch.cummax(x, dim=-1).values
    @classmethod
    def any(cls, x: Tensor) -> Tensor: return torch.max(x, dim=-1).values
    @classmethod
    def all(cls, x: Tensor) -> Tensor: return torch.min(x, dim=-1).values


class Product(Algebra):
    @classmethod
    def top(cls) -> Tensor: return torch.tensor([1.])
    @classmethod
    def bottom(cls) -> Tensor: return torch.tensor([0.])
    @classmethod
    def meet(cls, x: Tensor, y: Tensor) -> Tensor: return x * y
    @classmethod
    def join(cls, x: Tensor, y: Tensor) -> Tensor: return x + y - x * y
    @classmethod
    def neg(cls, x: Tensor) -> Tensor: return 1 - x

    @classmethod
    def running_meet(cls, x: Tensor) -> Tensor: return torch.cumprod(x, -1)
    @classmethod
    def all(cls, x: Tensor) -> Tensor: return torch.prod(x, -1)

class Lukasiewicz(Algebra):
    @classmethod
    def top(cls) -> Tensor: return torch.tensor([1.])
    @classmethod
    def bottom(cls) -> Tensor: return torch.tensor([0.])
    @classmethod
    def meet(cls, x: Tensor, y: Tensor) -> Tensor: return torch.clamp(x + y - 1, min=0.)
    @classmethod
    def join(cls, x: Tensor, y: Tensor) -> Tensor: return torch.clamp(x + y, max=1.)
    @classmethod
    def neg(cls, x: Tensor) -> Tensor: return 1 - x

class Hamacher(Algebra):
    @classmethod
    def top(cls) -> Tensor: return torch.tensor([1.])
    @classmethod
    def bottom(cls) -> Tensor: return torch.tensor([0.])
    @classmethod
    def meet(cls, x: Tensor, y: Tensor) -> Tensor: return torch.where((x==0) | (y==0), 0., (x + y) / (x + y - x * y))
    @classmethod
    def join(cls, x: Tensor, y: Tensor) -> Tensor: return (x + y) / (1 + x + y)
    @classmethod
    def neg(cls, x: Tensor) -> Tensor: return 1 - x
