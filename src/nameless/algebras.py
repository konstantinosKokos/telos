import torch
from torch import Tensor

from typing import Callable as Fn, Type
from abc import ABC, abstractmethod
from itertools import accumulate
from functools import reduce
from dataclasses import dataclass


def scan(fn: Fn[[Tensor, Tensor], Tensor], default: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return torch.stack(list(accumulate(x.unbind(-1), func=fn)), dim=-1)
    return f


def fold(fn: Fn[[Tensor, Tensor], Tensor], initial: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return reduce(fn, x.unbind(-1), initial)
    return f


@dataclass(eq=True)
class Properties:
    meet_commutative: bool
    join_commutative: bool
    meet_associative: bool
    join_associative: bool
    meet_idempotent: bool
    join_idempotent: bool
    absorption: bool
    distributivity: bool
    involution: bool
    de_morgan: bool
    complementarity: bool

    def __repr__(self) -> str:
        lines = [f"\t{name:<25}: {value}" for name, value in self.__dict__.items()]
        return "\n".join(lines)


class Algebra[T](ABC):
    dtype = T
    properties: Properties = ...

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


def commutative(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor) -> bool:
        return torch.allclose(fn(x, y), fn(y, x))
    return f

def associative(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor, Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor, z: Tensor) -> bool:
        return torch.allclose(fn(x, fn(y, z)), fn(fn(x, y), z))
    return f

def idempotent(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return torch.allclose(fn(x, x), x)
    return f


def check(algebra: Type[Algebra], test_size: int = 50) -> Properties:
    if algebra.dtype == bool:
        x, y, z = (torch.rand(test_size * 3, 1) > 0.5).chunk(3, dim=0)
    else:
        x, y, z = torch.rand(test_size * 3, 1).chunk(3, dim=0)

    meet, join, neg = algebra.meet, algebra.join, algebra.neg
    top, bottom = algebra.top(), algebra.bottom()

    meet_commutative = commutative(meet)(x, y)
    join_commutative = commutative(join)(x, y)
    meet_associative = associative(meet)(x, y, z)
    join_associative = associative(join)(x, y, z)
    meet_idempotent = idempotent(meet)(x)
    join_idempotent = idempotent(join)(x)
    absorption = all(
        (torch.allclose(x, meet(x, join(x, y))),
         torch.allclose(x, join(x, meet(x, y))))
    )
    distributivity = all(
        (torch.allclose(meet(x, join(y, z)), join(meet(x, y), meet(x, z))),
         torch.allclose(join(x, meet(y, z)), meet(join(x, y), join(x, z))))
    )
    involution = torch.allclose(x, neg(neg(x)))
    de_morgan = all(
        (torch.allclose(neg(join(x, y)), meet(neg(x), neg(y))),
         torch.allclose(neg(meet(x, y)), join(neg(x), neg(y))))
    )
    complementarity = all(
        (torch.allclose(meet(x, neg(x)), bottom),
         torch.allclose(join(x, neg(x)), top))
    )
    return Properties(
        meet_commutative=meet_commutative,
        join_commutative=join_commutative,
        meet_associative=meet_associative,
        join_associative=join_associative,
        meet_idempotent=meet_idempotent,
        join_idempotent=join_idempotent,
        absorption=absorption,
        distributivity=distributivity,
        involution=involution,
        de_morgan=de_morgan,
        complementarity=complementarity
    )


def algebra_factory(
        _dtype: Type[bool] | Type[float],
        top: Tensor,
        bottom: Tensor,
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor],
        neg: Fn[[Tensor], Tensor],
        running_meet: Fn[[Tensor], Tensor] | None = None,
        running_join: Fn[[Tensor], Tensor] | None = None,
        exists: Fn[[Tensor], Tensor] | None = None,
        forall: Fn[[Tensor], Tensor] | None = None,
) -> Type[Algebra]:

    if running_meet is None:
        running_meet = scan(meet, top)
    if running_join is None:
        running_join = scan(join, bottom)
    if exists is None:
        exists = fold(join, bottom)
    if forall is None:
        forall = fold(meet, top)

    class Instance(Algebra[_dtype], ABC):
        dtype = _dtype

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
    Instance.properties = check(Instance)
    return Instance


Boolean = algebra_factory(
    _dtype=bool,
    top=torch.tensor(True),
    bottom=torch.tensor(False),
    meet=torch.bitwise_and,
    join=torch.bitwise_or,
    neg=torch.bitwise_not,
    running_meet=lambda x: torch.cumprod(x, dim=-1),
    running_join=lambda x: torch.cummax(x, dim=-1).values,
    exists=lambda x: torch.any(x, dim=-1),
    forall=lambda x: torch.all(x, dim=-1)
)
Goedel = algebra_factory(
    _dtype=float,
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
    _dtype=float,
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=torch.mul,
    join=lambda x, y: x + y - x * y,
    neg=lambda x: 1 - x,
    running_meet=lambda x: torch.cumprod(x, dim=-1),
    forall=lambda x: torch.prod(x, -1),
)
Lukasiewicz = algebra_factory(
    _dtype=float,
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=lambda x, y: torch.clamp(x + y - 1, min=0.),
    join=lambda x, y: torch.clamp(x + y, max=1.),
    neg=lambda x: 1 - x
)
Hamacher = algebra_factory(
    _dtype=float,
    top=torch.tensor(1.),
    bottom=torch.tensor(0.),
    meet=lambda x, y: torch.where(torch.bitwise_or(torch.eq(x, 0), torch.eq(y, 0)) , 0., (x + y) / (x + y - x * y)),
    join=lambda x, y: (x + y) / ( 1 + x + y),
    neg=lambda x: 1 - x,
)