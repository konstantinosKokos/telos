from __future__ import annotations

from .syntax import (
    Formula, Variable, AbstractTop, AbstractBottom,
    Negation, Next, Disjunction, Conjunction, Implies, Until, free
)
from .algebras import Algebra

import torch
from torch import Tensor
from torch.nn import Module
from typing import Any, Callable as Fn
from torch.nn.functional import pad


class Trace:
    __match_args__ = ('names', 'values')

    def __init__(self, values: Tensor, names: tuple[str | Variable, ...]):
        self.names = tuple(Variable(n) if isinstance(n, str) else n for n in names)
        assert len(set(self.names)) == len(self.names), "Trace variable names must be unique."
        assert values.size(-2) == len(self.names), "Tensor's penultimate dim must match number of names."
        self.values = values

    def __getitem__(self, key: Variable | str) -> Tensor:
        if isinstance(key, str):
            key = Variable(key)
        return self.values[..., self.names.index(key), :]

    def __rshift__(self, other: Formula) -> Judgement:
        return Judgement(self, other)

    def __repr__(self) -> str:
        return f"Trace({', '.join(v.name for v in self.names)}, shape={tuple(self.values.shape)})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Trace)
            and self.names == other.names
            and torch.equal(self.values, other.values)
        )

    @property
    def variables(self) -> set[Variable]:
        return set(self.names)

    @property
    def duration(self) -> int:
        return self.values.size(-1)

    def bool(self, threshold: float = 0.5) -> Trace:
        return Trace(self.values > threshold, self.names)

    def vmap(self, fn: Fn[[Tensor], Tensor]) -> Trace:
        return Trace(torch.vmap(fn, in_dims=-2, out_dims=-2)(self.values), self.names)

    def to(self, device) -> Trace:
        return Trace(self.values.to(device), self.names)


def mkTrace(**vars: Tensor) -> Trace:
        assert vars, "Trace needs at least one variable."
        assert len({t.size() for t in vars.values()}) == 1, "Variables must share shape."
        return Trace(torch.stack(list(vars.values()), dim=-2), tuple(vars.keys()))


class Judgement:
    def __init__(self, trace: Trace, conclusion: Formula):
        assert free(conclusion).issubset(trace.variables)
        self.trace = trace
        self.conclusion = conclusion

    def __repr__(self) -> str:
        return f'{self.trace} ⊨? {self.conclusion}'

    def __eq__(self, other: Any):
        return isinstance(other, Judgement) and all((self.trace == other.trace, self.conclusion == other.conclusion))


class Model(Module):
    def __init__(self, algebra: Algebra):
        super().__init__()
        self.algebra = algebra

    def forward(self, judgement: Judgement, return_trajectory: bool = False) -> Tensor:
        algebra = self.algebra

        def go(trace: Trace, conclusion: Formula) -> Tensor:
            def reshape(result: Tensor) -> Tensor:
                return result.expand_as(trace.values[..., 0, :])

            match conclusion:
                case AbstractTop():
                    return reshape(algebra.top)
                case AbstractBottom():
                    return reshape(algebra.bottom)
                case Variable(_):
                    return trace[conclusion]
                case Negation(Until(AbstractTop(), Negation(x))):
                    return algebra.running_meet(go(trace, x).flip(-1)).flip(-1)
                case Negation(x):
                    return algebra.neg(go(trace, x))
                case Next(x):
                    return pad(go(trace, x)[..., 1:], pad=(0, 1), value=algebra.bottom)
                case Disjunction(l, r):
                    return algebra.join(go(trace, l), go(trace, r))
                case Conjunction(l, r):
                    return algebra.meet(go(trace, l), go(trace, r))
                case Implies(l, r):
                    return algebra.implies(go(trace, l), go(trace, r))
                case Until(AbstractTop(), r):
                    return algebra.running_join(go(trace, r).flip(-1)).flip(-1)
                case Until(l, r):
                    lss = algebra.span_meet(go(trace, l))
                    rs = go(trace, r)[..., None, :]
                    return algebra.exists(algebra.meet(lss, rs))
                case _:
                    raise ValueError

        result = go(judgement.trace, judgement.conclusion)
        return result if return_trajectory else result[..., 0]
