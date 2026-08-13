from __future__ import annotations

import pytest
import torch

from telos.algebras import (
    Algebra, Boltzmann, Goedel, Lukasiewicz, Mellowmax, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE, KleeneDienes,
)
from telos.algebras.gradients import (
    DURATIONS, Sampler, collapsing, ladder, nonnegative, normalized, regime, selective,
)

TRIALS = 512
DURATION = DURATIONS[0]


def uniform(lo: float, hi: float) -> Sampler:
    def f(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, dtype=torch.float64) * (hi - lo) + lo
    return f


unit, unit_near = uniform(1e-3, 1 - 1e-3), uniform(1 - 2 ** -5, 1 - 1e-6)
real, real_near = uniform(-10., 10.), uniform(5., 10.)

Profile = tuple[Algebra, tuple[Sampler, Sampler], str, bool, bool, bool]

profiles: list[Profile] = [
    (Goedel(), (unit, unit_near), 'dense', True, True, True),
    (KleeneDienes(), (unit, unit_near), 'dense', True, True, True),
    (Lukasiewicz(), (unit, unit_near), 'saturating', False, True, False),
    (Product(), (unit, unit_near), 'exponential', False, True, False),
    (Robustness(), (real, real_near), 'dense', True, True, True),
    (Frank(p=0.5, trainable=False), (unit, unit_near), 'exponential', False, True, False),
    (Hamacher(p=2., trainable=False), (unit, unit_near), 'exponential', False, True, False),
    (Yager(p=2., trainable=False), (unit, unit_near), 'saturating', False, True, False),
    (SchweizerSklar(p=2., trainable=False), (unit, unit_near), 'saturating', False, True, False),
    (AczelAlsina(p=2., trainable=False), (unit, unit_near), 'exponential', False, True, False),
    (Dombi(p=2., trainable=False), (unit, unit_near), 'polynomial', False, True, False),
    (SugenoWeber(p=1., trainable=False), (unit, unit_near), 'saturating', False, True, False),
    (LSE(p=2., trainable=False), (real, real_near), 'dense', False, True, True),
    (Boltzmann(beta=2., trainable=False), (real, real_near), 'dense', False, False, True),
    (Mellowmax(beta=2., trainable=False), (real, real_near), 'dense', False, True, True),
]
profiles = [(A.double(), samplers, *expected) for A, samplers, *expected in profiles]

params = pytest.mark.parametrize(
    'A, samplers, credit, is_selective, is_nonnegative, is_normalized',
    profiles,
    ids=[type(A).__name__ for A, *_ in profiles],
)


@pytest.fixture(autouse=True)
def seeded() -> None:
    torch.manual_seed(0)


def holds(A: Algebra, samplers: tuple[Sampler, ...], predicate) -> bool:
    return all(
        predicate(reduce)(sample((TRIALS, DURATION)))
        for reduce in (A.forall, A.exists)
        for sample in samplers
    )


@params
def test_credit_profile(
        A: Algebra,
        samplers: tuple[Sampler, Sampler],
        credit: str,
        is_selective: bool,
        is_nonnegative: bool,
        is_normalized: bool):
    wide, _ = samplers
    assert regime(DURATIONS, ladder(A.forall, wide, DURATIONS, TRIALS)) == credit


@params
def test_selectivity(
        A: Algebra,
        samplers: tuple[Sampler, Sampler],
        credit: str,
        is_selective: bool,
        is_nonnegative: bool,
        is_normalized: bool):
    assert holds(A, samplers, selective) == is_selective


@params
def test_nonnegativity(
        A: Algebra,
        samplers: tuple[Sampler, Sampler],
        credit: str,
        is_selective: bool,
        is_nonnegative: bool,
        is_normalized: bool):
    assert holds(A, samplers, nonnegative) == is_nonnegative


@params
def test_normalization(
        A: Algebra,
        samplers: tuple[Sampler, Sampler],
        credit: str,
        is_selective: bool,
        is_nonnegative: bool,
        is_normalized: bool):
    assert holds(A, samplers, normalized) == is_normalized


@params
def test_saturation_is_forward(
        A: Algebra,
        samplers: tuple[Sampler, Sampler],
        credit: str,
        is_selective: bool,
        is_nonnegative: bool,
        is_normalized: bool):
    wide, _ = samplers
    collapses = collapsing(A.forall, A.bottom)(wide((TRIALS, DURATION)))
    assert collapses == (credit == 'saturating')


def test_selectivity_is_orthogonal_to_density():
    assert {'dense'} == {credit for _, _, credit, sel, *_ in profiles if sel}


def test_boolean_admits_no_credit():
    with pytest.raises(RuntimeError):
        torch.zeros(TRIALS, DURATION, dtype=torch.bool).requires_grad_(True)
