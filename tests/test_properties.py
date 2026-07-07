from __future__ import annotations

import pytest
import torch

from telos.algebras import (
    Algebra, Boolean, Goedel, Lukasiewicz, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE,
    KleeneDienes,
)
from telos.algebras.properties import (
    commutative, associative, idempotent, absorption,
    distributive, involutive, de_morgan, complementary, residuated,
)

n = 50

laws = [
    ('meet_commutative', lambda A, x, y, z: commutative(A.meet)(x, y)),
    ('join_commutative', lambda A, x, y, z: commutative(A.join)(x, y)),
    ('meet_associative', lambda A, x, y, z: associative(A.meet)(x, y, z)),
    ('join_associative', lambda A, x, y, z: associative(A.join)(x, y, z)),
    ('meet_idempotent',  lambda A, x, y, z: idempotent(A.meet)(x)),
    ('join_idempotent',  lambda A, x, y, z: idempotent(A.join)(x)),
    ('absorption',       lambda A, x, y, z: absorption(A.meet, A.join)(x, y)),
    ('distributivity',   lambda A, x, y, z: distributive(A.meet, A.join)(x, y, z)),
    ('involutive',       lambda A, x, y, z: involutive(A.neg)(x)),
    ('de_morgan',        lambda A, x, y, z: de_morgan(A.meet, A.join, A.neg)(x, y)),
    ('complementarity',  lambda A, x, y, z: complementary(A.meet, A.join, A.neg, A.top, A.bottom)(x)),
    ('residuation',      lambda A, x, y, z: residuated(A.meet, A.implies)(x, y, z)),
]

strict = {'meet_idempotent', 'join_idempotent', 'absorption', 'distributivity', 'complementarity'}
nilpotent = {'meet_idempotent', 'join_idempotent', 'absorption', 'distributivity'}

instances: list[tuple[Algebra, set[str]]] = [
    (Boolean(), set()),
    (Goedel(), {'complementarity'}),
    (Lukasiewicz(), nilpotent),
    (Product(), strict),
    (Robustness(), {'complementarity'}),
    (Frank(p=0.5, trainable=False), strict),
    (Frank(p=2., trainable=False, upper=True), strict),
    (Hamacher(p=2., trainable=False), strict),
    (Yager(p=2., trainable=False), strict),
    (SchweizerSklar(p=2., trainable=False), nilpotent),
    (AczelAlsina(p=2., trainable=False), strict),
    (Dombi(p=2., trainable=False), strict),
    (SugenoWeber(p=1., trainable=False), strict),
    (LSE(p=2., trainable=False), strict),
    (KleeneDienes(), {'complementarity'}),
]


@pytest.fixture(scope='module')
def samples() -> dict[torch.dtype, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    fuzzy = torch.rand(3 * n, 1).chunk(3, dim=0)
    boolean = (torch.rand(3 * n, 1) > 0.5).chunk(3, dim=0)
    return {torch.float32: fuzzy, torch.bool: boolean}


def xyz(algebra: Algebra, samples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return samples[algebra.dtype]


@pytest.mark.parametrize('A, fails', instances, ids=[type(a).__name__ for a, _ in instances])
@pytest.mark.parametrize('name, predicate', laws, ids=[law[0] for law in laws])
def test_law(A, fails, samples, name, predicate):
    assert predicate(A, *xyz(A, samples)) == (name not in fails)
