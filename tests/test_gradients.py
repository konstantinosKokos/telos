from __future__ import annotations

import pytest
import torch

from telos import Variable, Model, mkTrace, eventually, always
from telos.algebras import Goedel, Lukasiewicz, Product, Robustness, Frank


algebras = [
    Goedel(),
    Lukasiewicz(),
    Product(),
    Robustness(),
    Frank(lam=0.5, trainable=False),
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


def test_frank_lam_trainable():
    A = Frank(lam=0.5, trainable=True)
    A.meet(fresh(time), fresh(time)).sum().backward()
    assert A._lam.grad is not None


def test_frank_lam_frozen():
    A = Frank(lam=0.5, trainable=False)
    A.meet(fresh(time), fresh(time)).sum().backward()
    assert A._lam.grad is None


def test_frank_lam_through_model():
    A = Frank(lam=0.5, trainable=True)
    Model(A)(mkTrace(p=fresh(time), q=fresh(time)) >> phi).backward()
    assert A._lam.grad is not None
