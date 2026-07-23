from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from telos import Model, mkTrace, eventually, always
from telos.syntax import (
    Formula, Variable, AbstractTop, AbstractBottom,
    Negation, Next, Disjunction, Conjunction, Implies, Until,
)
from telos.algebras import (
    Algebra, Boltzmann, Boolean, Goedel, Lukasiewicz, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE,
    KleeneDienes,
)


p, q = Variable('p'), Variable('q')
duration = 6


def sat(values: dict[Variable, np.ndarray], phi: Formula, t: int) -> bool:
    match phi:
        case AbstractTop(): return True
        case AbstractBottom(): return False
        case Variable(_): return bool(values[phi][t])
        case Negation(x): return not sat(values, x, t)
        case Next(x): return t + 1 < duration and sat(values, x, t + 1)
        case Disjunction(l, r): return sat(values, l, t) or sat(values, r, t)
        case Conjunction(l, r): return sat(values, l, t) and sat(values, r, t)
        case Implies(l, r): return not sat(values, l, t) or sat(values, r, t)
        case Until(l, r): return any(
            sat(values, r, u) and all(sat(values, l, v) for v in range(t, u + 1))
            for u in range(t, duration))
        case _: raise ValueError


formulas: st.SearchStrategy[Formula] = st.recursive(
    st.sampled_from((p, q, AbstractTop(), AbstractBottom())),
    lambda sub: st.one_of(
        sub.map(Negation),
        sub.map(Next),
        sub.map(eventually),
        sub.map(always),
        st.tuples(sub, sub).map(lambda lr: Conjunction(*lr)),
        st.tuples(sub, sub).map(lambda lr: Disjunction(*lr)),
        st.tuples(sub, sub).map(lambda lr: Implies(*lr)),
        st.tuples(sub, sub).map(lambda lr: Until(*lr)),
    ),
    max_leaves=8,
)


@given(phi=formulas, values=hnp.arrays(np.bool_, (2, duration)))
@settings(max_examples=300, deadline=None, derandomize=True)
def test_boolean_semantics(phi: Formula, values: np.ndarray):
    trace = mkTrace(p=torch.from_numpy(values[0]), q=torch.from_numpy(values[1]))
    trajectory = Model(Boolean())(trace >> phi, return_trajectory=True)
    assert trajectory.tolist() == [sat({p: values[0], q: values[1]}, phi, t) for t in range(duration)]


algebras: list[Algebra] = [
    Goedel(),
    Lukasiewicz(),
    Product(),
    Robustness(),
    Frank(p=0.5, trainable=False),
    Hamacher(p=2., trainable=False),
    Yager(p=2., trainable=False),
    SchweizerSklar(p=2., trainable=False),
    AczelAlsina(p=2., trainable=False),
    Dombi(p=2., trainable=False),
    SugenoWeber(p=1., trainable=False),
    LSE(p=2., trainable=False),
    KleeneDienes(),
    Boltzmann(beta=2., trainable=False),
]

phi = (Until(p, q) | ~Next(p)) & always(p > eventually(q))


def test_trajectory_prefix():
    judgement = mkTrace(p=torch.rand(duration), q=torch.rand(duration)) >> phi
    model = Model(Goedel())
    scalar, trajectory = model(judgement), model(judgement, return_trajectory=True)
    assert scalar.shape == ()
    assert trajectory.shape == (duration,)
    assert torch.equal(trajectory[..., 0], scalar)


@pytest.mark.parametrize('A', algebras, ids=lambda a: type(a).__name__)
def test_backward(A: Algebra):
    torch.manual_seed(0)
    tp, tq = (torch.rand(duration, requires_grad=True) for _ in range(2))
    Model(A)(mkTrace(p=tp, q=tq) >> phi).backward()
    assert tp.grad.isfinite().all() and tq.grad.isfinite().all()


parametric: list[tuple[type[Algebra], dict[str, float]]] = [
    (Frank, dict(p=0.5)),
    (Hamacher, dict(p=2.)),
    (Yager, dict(p=2.)),
    (SchweizerSklar, dict(p=2.)),
    (AczelAlsina, dict(p=2.)),
    (Dombi, dict(p=2.)),
    (SugenoWeber, dict(p=1.)),
    (LSE, dict(p=2.)),
    (Boltzmann, dict(beta=2.)),
]


@pytest.mark.parametrize('cls, kwargs', parametric, ids=[c.__name__ for c, _ in parametric])
def test_backward_parameters(cls: type[Algebra], kwargs: dict[str, float]):
    torch.manual_seed(0)
    A = cls(**kwargs, trainable=True)
    Model(A)(mkTrace(p=torch.rand(duration), q=torch.rand(duration)) >> phi).backward()
    assert all(parameter.grad is not None for parameter in A.parameters())
