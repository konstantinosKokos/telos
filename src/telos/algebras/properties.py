from __future__ import annotations
import torch
from torch import Tensor
from typing import Callable as Fn


def close(a: Tensor, b: Tensor) -> bool:
    return torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def commutative(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor) -> bool:
        return close(fn(x, y), fn(y, x))
    return f


def associative(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor, Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor, z: Tensor) -> bool:
        return close(fn(x, fn(y, z)), fn(fn(x, y), z))
    return f


def idempotent(fn: Fn[[Tensor, Tensor], Tensor]) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return close(fn(x, x), x)
    return f


def absorption(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor]
) -> Fn[[Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor) -> bool:
        return all(
            (close(x, meet(x, join(x, y))),
             close(x, join(x, meet(x, y))))
        )
    return f


def distributive(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor]
) -> Fn[[Tensor, Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor, z: Tensor) -> bool:
        return all(
            (close(meet(x, join(y, z)), join(meet(x, y), meet(x, z))),
             close(join(x, meet(y, z)), meet(join(x, y), join(x, z))))
        )
    return f


def involutive(neg: Fn[[Tensor], Tensor]) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return close(x, neg(neg(x)))
    return f


def de_morgan(
        meet: Fn[[Tensor, Tensor], Tensor],
        join: Fn[[Tensor, Tensor], Tensor],
        neg: Fn[[Tensor], Tensor]
) -> Fn[[Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor) -> bool:
        return all(
            (close(neg(join(x, y)), meet(neg(x), neg(y))),
             close(neg(meet(x, y)), join(neg(x), neg(y))))
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
            (close(meet(x, neg(x)), bottom),
             close(join(x, neg(x)), top))
        )
    return f


def residuated(
        meet: Fn[[Tensor, Tensor], Tensor],
        implies: Fn[[Tensor, Tensor], Tensor],
) -> Fn[[Tensor, Tensor, Tensor], bool]:
    def f(x: Tensor, y: Tensor, z: Tensor) -> bool:
        return close(implies(meet(x, y), z), implies(y, implies(x, z)))
    return f
