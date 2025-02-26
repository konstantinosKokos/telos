from __future__ import annotations

from .syntax import (
    Formula, Variable, AbstractTop, AbstractBottom,
    Negation, Next, Disjunction, Conjunction, Implies, Until, free
)
from .algebras import Algebra

import torch
from torch import Tensor
from typing import Any, Protocol
from torch.nn.functional import pad

from functools import lru_cache


class Trace(dict[Variable, Tensor]):
    def __init__(self, mapping: dict[Variable, Tensor]):
        assert len({v.size() for v in mapping.values()}) == 1
        super().__init__(mapping)

    @property
    def variables(self) -> set[Variable]:
        return set(self.keys())

    def __rshift__(self, other: Formula) -> Judgement:
        return Judgement(self, other)

    def __len__(self) -> int:
        return next(iter(self.values())).size(-1)

    def __eq__(self, other: Any) -> bool:
        return all(
            (isinstance(other, Trace),
             self.variables == other.variables,
             all(torch.all(self[k] == other[k]) for k in self.variables))
        )

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items())))


class Judgement:
    def __init__(self, trace: Trace, conclusion: Formula):
        assert free(conclusion).issubset(trace.variables)
        self.trace = trace
        self.conclusion = conclusion

    def __repr__(self) -> str:
        return f'{self.trace} ⊨? {self.conclusion}'

    def __eq__(self, other: Any):
        return isinstance(other, Judgement) and all((self.trace == other.trace, self.conclusion == other.conclusion))

    def __hash__(self) -> int:
        return hash(map(hash, (self.trace, self.conclusion)))


class Model(Protocol):
    def __call__(self, judgement: Judgement, return_trajectory: bool = False) -> Tensor: ...


def model(algebra: Algebra, cache_size: int = 128) -> Model:
    @lru_cache(maxsize=cache_size)
    def go(j: Judgement) -> Tensor:
        trace, conclusion = j.trace, j.conclusion

        def reshape(result: Tensor) -> Tensor:
            return result.expand_as(next(iter(trace.values())))

        match conclusion:
            case AbstractTop():
                return reshape(algebra.top)
            case AbstractBottom():
                return reshape(algebra.bottom)
            case Variable(x):
                return trace[Variable(x)]
            case Negation(x):
                return algebra.neg(go(trace >> x))
            case Next(x):
                return pad(go(trace >> x)[..., 1:], pad=(0, 1), value=algebra.bottom)
            case Disjunction(l, r):
                return algebra.join(go(trace >> l), go(trace >> r))
            case Conjunction(l, r):
                return algebra.meet(go(trace >> l), go(trace >> r))
            case Implies(l, r):
                return algebra.implies(go(trace >> l), go(trace >> r))
            case Until(l, r):
                lss = algebra.span_meet(go(trace >> l))
                rs = go(trace >> r)[..., None, :]
                return algebra.exists(algebra.meet(lss, rs))
            case _:
                raise ValueError

    def evaluate(judgement: Judgement, return_trajectory: bool = False) -> Tensor:
        result = go(judgement)
        return result if return_trajectory else result[..., 0]

    return evaluate
