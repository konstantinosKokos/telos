from __future__ import annotations
import math
import torch
from torch import Tensor
from typing import Callable as Fn, Sequence

Reduction = Fn[[Tensor], Tensor]
Sampler = Fn[[tuple[int, ...]], Tensor]

DURATIONS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)


def shares(reduce: Reduction) -> Reduction:
    def f(x: Tensor) -> Tensor:
        x = x.detach().requires_grad_(True)
        return torch.autograd.grad(reduce(x).sum(), x)[0]
    return f


def nonnegative(reduce: Reduction) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return bool((shares(reduce)(x) >= -1e-9).all())
    return f


def normalized(reduce: Reduction) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return bool(((shares(reduce)(x).sum(dim=-1) - 1).abs() < 1e-6).all())
    return f


def selective(reduce: Reduction) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        g = shares(reduce)(x).abs()
        return bool(((g > 1e-3 * g.amax(dim=-1, keepdim=True)).sum(dim=-1) == 1).all())
    return f


def mass(reduce: Reduction) -> Fn[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        return shares(reduce)(x).abs().sum(dim=-1).mean()
    return f


def collapsing(reduce: Reduction, bottom: Tensor) -> Fn[[Tensor], bool]:
    def f(x: Tensor) -> bool:
        return bool((reduce(x) == bottom).any())
    return f


def ladder(reduce: Reduction, sample: Sampler, durations: Sequence[int], trials: int) -> list[float]:
    return [float(mass(reduce)(sample((trials, n)))) for n in durations]


def exponents(durations: Sequence[int], masses: Sequence[float]) -> list[float | None]:
    rungs = list(zip(durations, masses))
    return [
        -math.log(b / a) / math.log(tb / ta) if a > 0 and b > 0 else None
        for (ta, a), (tb, b) in zip(rungs, rungs[1:])
    ]


def extinguished(masses: Sequence[float]) -> bool:
    return any(m == 0. for m in masses)


def drift(ks: Sequence[float | None]) -> float:
    live = [k for k in ks if k is not None]
    return max(live) / max(live[0], 1e-9) if live else float('nan')


def tail(ks: Sequence[float | None]) -> float:
    live = [k for k in ks if k is not None]
    return live[-1] if live else float('nan')


def regime(
        durations: Sequence[int],
        masses: Sequence[float],
        tol: float = 1e-2,
        drift_bound: float = 3.
) -> str:
    if extinguished(masses):
        return 'saturating'
    ks = exponents(durations, masses)
    if tail(ks) <= tol:
        return 'dense'
    return 'polynomial' if drift(ks) < drift_bound else 'exponential'
