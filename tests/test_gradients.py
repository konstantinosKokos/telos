from __future__ import annotations

import pytest
import torch

from telos import Variable, Model, mkTrace, eventually, always
from telos.algebras import (
    Goedel, Lukasiewicz, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE,
)


algebras = [
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
]
time = 8

p, q = Variable('p'), Variable('q')
phi = eventually(q) & always(p | q)


def fresh(*shape: int) -> torch.Tensor:
    return torch.rand(*shape, requires_grad=True)


@pytest.mark.parametrize('A', algebras, ids=lambda a: type(a).__name__)
class TestOperatorGradients:
    @pytest.mark.parametrize('op', ['meet', 'join', 'implies'])
    def test_binary(self, A, op):
        x, y = fresh(time), fresh(time)
        getattr(A, op)(x, y).sum().backward()
        if isinstance(A, Goedel) and op == 'implies':
            assert x.grad is None and y.grad is not None
        else:
            assert x.grad is not None and y.grad is not None

    def test_neg(self, A):
        x = fresh(time)
        A.neg(x).sum().backward()
        assert x.grad is not None


@pytest.mark.parametrize('A', algebras, ids=lambda a: type(a).__name__)
@pytest.mark.parametrize('shape', [(time,), (4, time)], ids=['unbatched', 'batched'])
def test_evaluator(A, shape):
    tp, tq = fresh(*shape), fresh(*shape)
    out = Model(A)(mkTrace(p=tp, q=tq) >> phi)
    assert out.shape == shape[:-1]
    out.sum().backward()
    assert tp.grad is not None and tq.grad is not None


def test_evaluator_trajectory():
    judgement = mkTrace(p=torch.rand(time), q=torch.rand(time)) >> phi
    m = Model(Goedel())
    scalar = m(judgement)
    trajectory = m(judgement, return_trajectory=True)
    assert scalar.shape == ()
    assert trajectory.shape == (time,)
    assert torch.equal(trajectory[..., 0], scalar)


parametric = [
    (Frank, dict(p=0.5)),
    (Hamacher, dict(p=2.)),
    (Yager, dict(p=2.)),
    (SchweizerSklar, dict(p=2.)),
    (AczelAlsina, dict(p=2.)),
    (Dombi, dict(p=2.)),
    (SugenoWeber, dict(p=1.)),
    (LSE, dict(p=2.)),
]


@pytest.mark.parametrize('cls, kwargs', parametric, ids=[c.__name__ for c, _ in parametric])
def test_param_trainable(cls, kwargs):
    A = cls(**kwargs, trainable=True)
    A.meet(fresh(time), fresh(time)).sum().backward()
    assert all(p.grad is not None for p in A.parameters())


@pytest.mark.parametrize('cls, kwargs', parametric, ids=[c.__name__ for c, _ in parametric])
def test_param_frozen(cls, kwargs):
    A = cls(**kwargs, trainable=False)
    A.meet(fresh(time), fresh(time)).sum().backward()
    assert all(p.grad is None for p in A.parameters())


@pytest.mark.parametrize('cls, kwargs', parametric, ids=[c.__name__ for c, _ in parametric])
def test_param_through_model(cls, kwargs):
    A = cls(**kwargs, trainable=True)
    Model(A)(mkTrace(p=fresh(time), q=fresh(time)) >> phi).backward()
    assert all(p.grad is not None for p in A.parameters())
