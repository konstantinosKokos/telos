from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable as Fn
from itertools import accumulate
from functools import reduce, lru_cache


def scan(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return torch.stack(list(accumulate(x.unbind(-1), func=fn)), dim=-1)
    return f


def fold(fn: Fn[[Tensor, Tensor], Tensor], initial: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return reduce(fn, x.unbind(-1), initial)
    return f


def span_dynamic(
        fn: Fn[[Tensor, Tensor], Tensor],
        initial: Tensor,
        cache_size: int | None = None
) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        n = x.size(-1)

        @lru_cache(maxsize=cache_size)
        def span_result(i: int, j: int) -> Tensor:
            if i == j:
                return x[..., i]
            split_idx = i + (j - i + 1) // 2
            return fn(span_result(i, split_idx - 1), span_result(split_idx, j))
        return torch.stack(
            [torch.stack(
                [span_result(i, j) if i <= j else initial.expand_as(x[..., 0]) for j in range(n)], dim=-1) for i in range(n)
            ],
            dim=-2
        )
    return f


def span_vectorized(fn: [[Tensor], Tensor], initial: Tensor) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        n = x.size(-1)
        mask = torch.triu(torch.ones(n, n, dtype=x.dtype, device=x.device)).bool()
        return fn(torch.where(mask, x[..., None, :], initial)) * mask
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

    def span_meet(self, x: Tensor) -> Tensor:
        if self.properties.meet_associative and self.properties.meet_commutative:
            return span_vectorized(self.running_meet, self.top)(x)
        return span_dynamic(self.meet, self.bottom)(x)

    def span_join(self, x: Tensor) -> Tensor:
        if self.properties.join_associative and self.properties.join_commutative:
            return span_vectorized(self.running_join, self.bottom)(x)
        return self.span_join_dynamic(x)



class FuzzyBase(Algebra, ABC):
    dtype = float

    def __init__(self):
        super().__init__()
        self.register_buffer('_top', torch.tensor(1.))
        self.register_buffer('_bottom', torch.tensor(0.))

    def neg(self, x: Tensor) -> Tensor:
        return 1 - x


@dataclass(eq=True)
class Properties:
    meet_commutative:   bool  # x ∧ y = y ∧ x
    join_commutative:   bool  # x ∨ y = y ∨ x
    meet_associative:   bool  # x ∧ (y ∧ z) = (x ∧ y) ∧ z
    join_associative:   bool  # x ∨ (y ∨ z) = (x ∨ y) ∨ z
    meet_idempotent:    bool  # x ∧ x = x
    join_idempotent:    bool  # x ∨ x = x
    absorption:         bool  # x ∧ (x ∨ y) = x ∨ (x ∧ y)
    distributivity:     bool  # {x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)} ⋀ {x ∨ (y ∧ z) = (x ∨ y) ∧ (x ∨ z)}
    involutive:         bool  # ¬(¬x) = x
    de_morgan:          bool  # {¬(x ∨ y) = ¬x ∧ ¬y} ⋀ {¬(x ∧ y) = (¬x ∨ ¬y)}
    complementarity:    bool  # {x ∧ ¬x = ⊥} ⋀ (x ∨ ¬x = ⊤)
    residuation:        bool  # (x ∧ y) → z = y → (x → z)

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

        return Properties(
            meet_commutative=commutative(meet)(x, y),
            join_commutative=commutative(join)(x, y),
            meet_associative=associative(meet)(x, y, z),
            join_associative=associative(join)(x, y, z),
            absorption=absorption(meet, join)(x, y),
            meet_idempotent=idempotent(meet)(x),
            join_idempotent=idempotent(join)(x),
            distributivity=distributive(meet, join)(x, y, z),
            involutive=involutive(neg)(x),
            de_morgan=de_morgan(meet, join, neg)(x, y),
            complementarity=complementary(meet, join, neg, top, bottom)(x),
            residuation=residuated(meet, implies)(x, y, z)
        )


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

def absorption(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor]
) -> Fn[[Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor) -> bool:
        return all(
            (torch.allclose(x, meet(x, join(x, y))),
             torch.allclose(x, join(x, meet(x, y))))
        )
    return f

def distributive(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor]
) -> Fn[[Tensor, Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor, z: Tensor) -> bool:
        return all(
            (torch.allclose(meet(x, join(y, z)), join(meet(x, y), meet(x, z))),
             torch.allclose(join(x, meet(y, z)), meet(join(x, y), join(x, z))))
        )
    return f

def involutive(neg: Fn[[Tensor], Tensor]) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return torch.allclose(x, neg(neg(x)))
    return f

def de_morgan(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor],
        neg: Fn[[Tensor], Tensor]
) -> Fn[[Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor) -> bool:
        return all(
            (torch.allclose(neg(join(x, y)), meet(neg(x), neg(y))),
             torch.allclose(neg(meet(x, y)), join(neg(x), neg(y))))
        )
    return f

def complementary(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor],
        neg: Fn[[Tensor], Tensor],
        top: Tensor,
        bottom: Tensor
) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return all(
            (torch.allclose(meet(x, neg(x)), bottom),
             torch.allclose(join(x, neg(x)), top))
        )
    return f

def residuated(
        meet: Fn[[Tensor, Tensor], Tensor],
        implies: Fn[[Tensor, Tensor], Tensor],
) -> Fn[[Tensor, Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor, z: Tensor) -> bool:
        return torch.all(
            torch.all(implies(meet(x, y), z) == (implies(y, implies(x, z))))
        ).item()
    return f
