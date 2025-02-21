from .syntax import (
    Formula, Variable, AbstractTop, AbstractBottom,
    Negation, Next, Disjunction, Conjunction, Implies, Until, free
)
from .algebras import Algebra
import torch
from torch import Tensor
from typing import Callable as Fn

from functools import lru_cache


class Trace(dict[Variable, Tensor]):
    def __init__(self, mapping: dict[Variable, Tensor]):
        assert len({v.size(-1) for v in mapping.values()}) == 1
        super().__init__(mapping)

    @property
    def variables(self) -> set[Variable]:
        return set(self.keys())

    def __len__(self) -> int:
        return next(iter(self.values())).size(-1)


class Judgement:
    def __init__(self, trace: Trace, conclusion: Formula):
        assert free(conclusion).issubset(trace.variables)
        self.trace = trace
        self.conclusion = conclusion

    def __repr__(self) -> str:
        return f'{self.trace} ⊨? {self.conclusion}'


def suffix(trace: Trace, i: int) -> Trace:
    return Trace({var: vals[..., i:] for var, vals in trace.items()})


def subexprs(f: Formula) -> set[Formula]:
    match f:
        case AbstractTop() | AbstractBottom() | Variable(_): return {f}
        case Next(x) | Negation(x): return subexprs(x) | {f}
        case Disjunction(l, r) | Until(l, r): return subexprs(l) | subexprs(r) | {f}
        case _: raise ValueError


def model(algebra: Algebra, cache_size: int = 128) -> Fn[[Judgement], Tensor]:
    @lru_cache(maxsize=cache_size)
    def evaluate(j: Judgement) -> Tensor:
        match j.conclusion:
            case AbstractTop():
                return algebra.top
            case AbstractBottom():
                return algebra.bottom
            case Variable(x):
                #try:
                return j.trace[Variable(x)][..., 0]
                # except IndexError:
                #     return j.trace[Variable(x)].squeeze(-1)
            case Negation(x):
                return algebra.neg(evaluate(Judgement(j.trace, x)))
            case Next(x):
                return evaluate(Judgement(suffix(j.trace, 1), x))
            case Disjunction(l, r):
                r_ = evaluate(Judgement(j.trace, r))
                l_ = evaluate(Judgement(j.trace, l))
                return algebra.join(l_, r_)
            case Conjunction(l, r):
                r_ = evaluate(Judgement(j.trace, r))
                l_ = evaluate(Judgement(j.trace, l))
                return algebra.meet(l_, r_)
            case Implies(l, r):
                return algebra.implies(evaluate(Judgement(j.trace, l)), evaluate(Judgement(j.trace, r)))
            case Until(l, r):
                # TODO: debug this, suffix does not check the first element in the trace, 
                # and checks one extra element in the end
                # (i.e. returns a tensor [batch_size, 0] in the last iteration)
                rs, ls = [], []
                for i in range(len(j.trace)):
                    s_ = suffix(j.trace, i)
                    jud_r = Judgement(s_, r)
                    jud_l = Judgement(s_, l)
                    ev_r = evaluate(jud_r)
                    ev_l = evaluate(jud_l)
                    rs.append(ev_r)
                    ls.append(ev_l)
                rs = torch.stack(rs, dim=-1)
                ls = torch.stack(ls, dim=-1)
                # rs = torch.stack([evaluate(Judgement(suffix(j.trace, i), r)) for i in range(len(j.trace))], dim=-1)
                # ls = torch.stack([evaluate(Judgement(suffix(j.trace, i), l)) for i in range(len(j.trace))], dim=-1)
                return algebra.exists(algebra.meet(algebra.running_meet(ls), rs))
            case _:
                raise ValueError
    return evaluate
