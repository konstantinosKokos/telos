from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


class Algebra(ABC, Module):
    dtype = ...
    properties: Properties = ...

    @property
    def top(self) -> Tensor: return self._top
    @property
    def bottom(self) -> Tensor: return self._bottom
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
    residuation: bool

    def __repr__(self) -> str:
        lines = [f"\t{name:<25}: {value}" for name, value in self.__dict__.items()]
        return "\n".join(lines)

    @staticmethod
    def check(algebra: Algebra, test_size: int = 50) -> Properties:
        algebra = algebra.cpu()
        if algebra.dtype == bool:
            x, y, z = (torch.rand(test_size * 3, 1) > 0.5).chunk(3, dim=0)
        else:
            x, y, z = torch.rand(test_size * 3, 1).chunk(3, dim=0)

        top, bottom = algebra.top, algebra.bottom
        meet, join, implies, neg = algebra.meet, algebra.join, algebra.implies, algebra.neg

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
        residuation = torch.all(
            (torch.all((algebra.meet(x, y) <= z) == (y <= algebra.implies(x, z))))
        ).item()
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
            complementarity=complementarity,
            residuation=residuation
        )
