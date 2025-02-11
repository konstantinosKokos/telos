import torch
from torch import Tensor

from typing import Callable as Fn, Type
from abc import ABC, abstractmethod
from itertools import accumulate
from functools import reduce

def scan(fn: Fn[[Tensor, Tensor], Tensor], default: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return torch.stack(list(accumulate(x.unbind(-1), func=fn)), dim=-1)
    return f


def fold(fn: Fn[[Tensor, Tensor], Tensor], initial: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return reduce(fn, x.unbind(-1), initial)
    return f


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
    @abstractmethod
    def running_meet(cls, x: Tensor) -> Tensor: ...
    @classmethod
    @abstractmethod
    def running_join(cls, x: Tensor) -> Tensor: ...
    @classmethod
    @abstractmethod
    def exists(cls, x: Tensor) -> Tensor: ...
    @classmethod
    @abstractmethod
    def forall(cls, x: Tensor) -> Tensor: ...
    
    
def algebra_factory(
        top: Tensor,
        bottom: Tensor,
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor],
        neg: Fn[[Tensor], Tensor],
        running_meet: Fn[[Tensor], Tensor] | None = None,
        running_join: Fn[[Tensor], Tensor] | None = None,
        exists: Fn[[Tensor], Tensor] | None = None,
        forall: Fn[[Tensor], Tensor] | None = None
) -> Type[Algebra]:

    if running_meet is None:
        running_meet = scan(meet, top)
    if running_join is None:
        running_join = scan(join, bottom)
    if exists is None:
        exists = fold(join, bottom)
    if forall is None:
        forall = fold(meet, top)

    class Instance(Algebra, ABC):
        @classmethod
        def top(cls) -> Tensor: return top
        @classmethod
        def bottom(cls) -> Tensor: return bottom
        @classmethod
        def meet(cls, x: Tensor, y: Tensor) -> Tensor: return meet(x, y)
        @classmethod
        def join(cls, x: Tensor, y: Tensor) -> Tensor: return join(x, y)
        @classmethod
        def neg(cls, x: Tensor) -> Tensor: return neg(x)
        @classmethod
        def running_meet(cls, x: Tensor) -> Tensor: return running_meet(x)
        @classmethod
        def running_join(cls, x: Tensor) -> Tensor: return running_join(x)
        @classmethod
        def exists(cls, x: Tensor) -> Tensor: return exists(x)
        @classmethod
        def forall(cls, x: Tensor) -> Tensor: return forall(x)
    return Instance


Boolean = algebra_factory(
    top=torch.tensor([True]),
    bottom=torch.tensor([False]),
    meet=torch.bitwise_and,
    join=torch.bitwise_or,
    neg=torch.bitwise_not,
    running_meet=lambda x: torch.cumprod(x, dim=-1),
    running_join=lambda x: torch.cummax(x, dim=-1).values,
    exists=lambda x: torch.any(x, dim=-1),
    forall=lambda x: torch.all(x, dim=-1)
)
Goedel = algebra_factory(
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=torch.minimum,
    join=torch.maximum,
    neg=lambda x: 1 - x,
    running_meet=lambda x: torch.cummin(x, dim=-1).values,
    running_join=lambda x: torch.cummax(x, dim=-1).values,
    exists=lambda x: torch.max(x, dim=-1).values,
    forall=lambda x: torch.min(x, dim=-1).values
)
Product = algebra_factory(
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=torch.mul,
    join=lambda x, y: x + y - x * y,
    neg=lambda x: 1 - x,
    running_meet=lambda x: torch.cumprod(x, dim=-1),
    forall=lambda x: torch.prod(x, -1)[..., None]
)
Lukasiewicz = algebra_factory(
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=lambda x, y: torch.clamp(x + y - 1, min=0.),
    join=lambda x, y: torch.clamp(x + y, max=1.),
    neg=lambda x: 1 - x
)
Hamacher = algebra_factory(
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=lambda x, y: torch.where(torch.bitwise_or(torch.eq(x, 0), torch.eq(y, 0)) , 0., (x + y) / (x + y - x * y)),
    join=lambda x, y: (x + y) / ( 1 + x + y),
    neg=lambda x: 1 - x,
)