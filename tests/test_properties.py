from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from typing import Callable as Fn
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from telos.algebras import (
    Algebra, Boltzmann, Boolean, Goedel, Lifted, Lukasiewicz, Mellowmax, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE,
    KleeneDienes,
)
from telos.algebras.properties import (
    close, commutative, associative, idempotent, absorption, distributive,
    involutive, de_morgan, complementary, residuated, adjoint, monotone, unital, zero_free,
)

SHAPE = (8,)


def interval(lo: float, hi: float) -> st.SearchStrategy[torch.Tensor]:
    return hnp.arrays(
        np.float64,
        SHAPE,
        elements=st.floats(lo, hi, allow_subnormal=False)
    ).map(torch.from_numpy)


booleans = hnp.arrays(np.bool_, SHAPE).map(torch.from_numpy)


laws = [
    ('meet_commutative', lambda A, x, y, *_: commutative(A.meet)(x, y)),
    ('join_commutative', lambda A, x, y, *_: commutative(A.join)(x, y)),
    ('meet_associative', lambda A, x, y, z: associative(A.meet)(x, y, z)),
    ('join_associative', lambda A, x, y, z: associative(A.join)(x, y, z)),
    ('meet_idempotent',  lambda A, x, *_: idempotent(A.meet)(x)),
    ('join_idempotent',  lambda A, x, *_: idempotent(A.join)(x)),
    ('absorption',       lambda A, x, y, *_: absorption(A.meet, A.join)(x, y)),
    ('distributivity',   lambda A, x, y, z: distributive(A.meet, A.join)(x, y, z)),
    ('involutive',       lambda A, x, *_: involutive(A.neg)(x)),
    ('de_morgan',        lambda A, x, y, *_: de_morgan(A.meet, A.join, A.neg)(x, y)),
    ('complementarity',  lambda A, x, *_: complementary(A.meet, A.join, A.neg, A.top, A.bottom)(x)),
    ('residuation',      lambda A, x, y, z: residuated(A.meet, A.implies)(x, y, z)),
    ('adjunction',       lambda A, x, y, z: adjoint(A.meet, A.implies)(x, y, z)),
    ('monotone',         lambda A, x, y, z: monotone(A.meet, A.join, A.neg)(x, y, z)),
    ('unital',           lambda A, x, *_: unital(A.meet, A.join, A.top, A.bottom)(x)),
    ('zero_free',        lambda A, x, y, *_: zero_free(A.meet, A.join, A.top, A.bottom)(x, y)),
]

strict = {'meet_idempotent', 'join_idempotent', 'absorption', 'distributivity', 'complementarity'}
nilpotent = {'meet_idempotent', 'join_idempotent', 'absorption', 'distributivity', 'zero_free'}

instances: list[tuple[Algebra, set[str], st.SearchStrategy[torch.Tensor]]] = [
    (Boolean(), set(), booleans),
    (Goedel(), {'complementarity'}, interval(0., 1.)),
    (Lukasiewicz(), nilpotent, interval(0., 1.)),
    (Product(), strict, interval(1e-3, 1 - 1e-3)),
    (Robustness(), {'complementarity', 'adjunction'}, interval(-1e30, 1e30)),
    (Frank(p=0.5, trainable=False), strict, interval(1e-3, 1 - 1e-3)),
    (Frank(p=2., trainable=False, upper=True), strict, interval(1e-3, 1 - 1e-3)),
    (Hamacher(p=2., trainable=False), strict, interval(1e-3, 1 - 1e-3)),
    (Yager(p=2., trainable=False), nilpotent | {'complementarity'}, interval(0., 1.)),
    (SchweizerSklar(p=2., trainable=False), nilpotent, interval(0., 1.)),
    (AczelAlsina(p=2., trainable=False), strict, interval(1e-3, 1 - 1e-3)),
    (Dombi(p=2., trainable=False), strict, interval(1e-3, 1 - 1e-3)),
    (SugenoWeber(p=1., trainable=False), nilpotent | {'complementarity'}, interval(0., 1.)),
    (LSE(p=2., trainable=False), strict | {'adjunction'}, interval(-1e30, 1e30)),
    (KleeneDienes(), {'complementarity', 'adjunction'}, interval(0., 1.)),
    (Boltzmann(beta=2., trainable=False),
     {'meet_associative', 'join_associative', 'absorption', 'distributivity',
      'complementarity', 'residuation', 'monotone', 'adjunction'},
     interval(-10., 10.)),
    (Mellowmax(beta=2., trainable=False),
     {'meet_associative', 'join_associative', 'absorption', 'distributivity',
      'complementarity', 'residuation', 'adjunction'},
     interval(-10., 10.)),
]
instances = [(A.double(), fails, carrier) for A, fails, carrier in instances]

params = pytest.mark.parametrize(
    'A, fails, carrier, name, predicate',
    [(A, fails, carrier, name, predicate) for A, fails, carrier in instances for name, predicate in laws],
    ids=[f'{type(A).__name__}-{name}' for A, _, _ in instances for name, _ in laws],
)


@params
@given(data=st.data())
@settings(max_examples=100, deadline=None, derandomize=True)
def test_law_holds(
        A: Algebra,
        fails: set[str],
        carrier: st.SearchStrategy[torch.Tensor],
        name: str,
        predicate: Fn[[Algebra, Tensor, Tensor, Tensor], bool],
        data: st.DataObject):
    if name in fails:
        return
    x, y, z = (data.draw(carrier, label=l) for l in 'xyz')
    assert predicate(A, x, y, z)


witnessed: set[tuple[str, str]] = set()


@params
@settings(max_examples=200, deadline=None, derandomize=True)
@given(data=st.data())
def test_law_fails(
        A: Algebra,
        fails: set[str],
        carrier: st.SearchStrategy[torch.Tensor],
        name: str,
        predicate: Fn[[Algebra, Tensor, Tensor, Tensor], bool],
        data: st.DataObject):
    if name not in fails:
        return
    x, y, z = (data.draw(carrier, label=l) for l in 'xyz')
    if not predicate(A, x, y, z):
        witnessed.add((type(A).__name__, name))


def test_exemptions_witnessed():
    expected = {(type(A).__name__, name) for A, fails, _ in instances for name in fails}
    assert witnessed >= expected, sorted(expected - witnessed)


def observed[S](A: Lifted[S], s: S, probe: S) -> tuple[Tensor, Tensor]:
    return A.readout(s), A.readout(s.combine(probe))


def states_close[S](A: Lifted[S], a: S, b: S, probe: S) -> bool:
    return all(map(close, observed(A, a, probe), observed(A, b, probe)))


def combine_associative(A: Lifted, x: Tensor, y: Tensor, z: Tensor, w: Tensor) -> bool:
    a, b, c = A.embed(x), A.embed(y), A.embed(z)
    return states_close(A, a.combine(b).combine(c), a.combine(b.combine(c)), A.embed(w))


def neutral_identity(A: Lifted, x: Tensor, y: Tensor, z: Tensor, *_: Tensor) -> bool:
    s, probe = A.embed(x).combine(A.embed(y)), A.embed(z)
    return all(
        (states_close(A, s.neutral().combine(s), s, probe),
         states_close(A, s.combine(s.neutral()), s, probe))
    )


def section(A: Lifted, x: Tensor, *_: Tensor) -> bool:
    return close(A.readout(A.embed(x)), x)


state_laws = [
    ('combine_associative', combine_associative),
    ('neutral_identity', neutral_identity),
    ('section', section),
]

lifted_instances: list[tuple[Lifted, st.SearchStrategy[torch.Tensor]]] = [
    (Boltzmann(beta=2., trainable=False).double(), interval(-10., 10.)),
    (Mellowmax(beta=2., trainable=False).double(), interval(-10., 10.)),
]


state_params = pytest.mark.parametrize(
    'A, carrier, _, predicate',
    [(A, carrier, name, predicate) for A, carrier in lifted_instances for name, predicate in state_laws],
    ids=[f'{type(A).__name__}-{name}' for A, _ in lifted_instances for name, _ in state_laws],
)


@pytest.mark.parametrize('A', [A for A, _ in lifted_instances], ids=lambda a: type(a).__name__)
def test_neutral_readout_finite(A: Lifted):
    assert A.readout(A.embed(A.top).neutral()).isfinite().all()


def test_lifted_licenses():
    assert all(not ({'involutive', 'de_morgan'} & fails) for A, fails, _ in instances if isinstance(A, Lifted))


@state_params
@given(data=st.data())
@settings(max_examples=100, deadline=None, derandomize=True)
def test_state_law(
        A: Lifted,
        carrier: st.SearchStrategy[torch.Tensor],
        _: str,
        predicate: Fn[[Lifted, Tensor, Tensor, Tensor, Tensor], bool],
        data: st.DataObject):
    x, y, z, w = (data.draw(carrier, label=l) for l in 'xyzw')
    assert predicate(A, x, y, z, w)
