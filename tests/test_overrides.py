from __future__ import annotations

import pytest
import torch
from torch import Tensor
from typing import Callable as Fn
from functools import reduce
from itertools import accumulate

from telos.algebras import (
    Archimedean, Boltzmann, Boolean, Goedel, Lifted, Lukasiewicz, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE,
    KleeneDienes, State, TensorAlgebra,
)
from telos.algebras.base import scan, fold, span
from telos.algebras.properties import generated


algebras: list[TensorAlgebra] = [
    Boolean(),
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
]
batch, time = 4, 12

checks: list[tuple[str, Fn[[TensorAlgebra], Fn[[Tensor], Tensor]]]] = [
    ('running_meet', lambda A: scan(A.meet)),
    ('running_join', lambda A: scan(A.join)),
    ('forall',       lambda A: fold(A.meet, A.top)),
    ('exists',       lambda A: fold(A.join, A.bottom)),
    ('span_meet',    lambda A: span(scan(A.meet), A.top, A.bottom)),
    ('span_join',    lambda A: span(scan(A.join), A.bottom, A.bottom)),
]


@pytest.fixture(scope='module')
def samples() -> dict[torch.dtype, Tensor]:
    torch.manual_seed(0)
    return {
        torch.float32: torch.rand(batch, time),
        torch.bool: torch.rand(batch, time) > 0.5,
    }


def close(a: Tensor, b: Tensor) -> bool:
    return torch.equal(a, b) if a.dtype == torch.bool else torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def grad(out: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(out.sum(), x)[0]


@pytest.mark.parametrize('A', algebras, ids=lambda a: type(a).__name__)
@pytest.mark.parametrize('name, reference', checks, ids=[c[0] for c in checks])
def test_override(
        A: TensorAlgebra,
        samples: dict[torch.dtype, Tensor],
        name: str,
        reference: Fn[[TensorAlgebra], Fn[[Tensor], Tensor]]):
    x = samples[A.dtype].clone().requires_grad_(A.dtype.is_floating_point)
    override, spec = getattr(A, name)(x), reference(A)(x)
    assert close(override, spec)
    if x.requires_grad:
        assert close(grad(override, x), grad(spec, x))


@pytest.mark.parametrize('A', [A for A in algebras if isinstance(A, Archimedean)], ids=lambda a: type(a).__name__)
def test_generated(A: Archimedean, samples: dict[torch.dtype, Tensor]):
    x = samples[torch.float32]
    assert generated(A.meet, A.g, A.g_inv)(x, x.flip(0))


lifted: list[Lifted] = [
    Boltzmann(beta=2., trainable=False),
]


def ticks[S: State](A: Lifted[S], states: S) -> list[S]:
    return [A.fmap(states, lambda c, i=i: c[..., i]) for i in range(states.duration)]


lifted_checks: list[tuple[str, str]] = [
    ('running_meet', 'embed_meet'),
    ('running_join', 'embed_join'),
]


@pytest.mark.parametrize('A', lifted, ids=lambda a: type(a).__name__)
@pytest.mark.parametrize('name, side', lifted_checks, ids=[c[0] for c in lifted_checks])
def test_lifted_override(A: Lifted, name: str, side: str):
    torch.manual_seed(0)
    x = (torch.rand(batch, time) * 20 - 10).requires_grad_(True)
    override = A.readout(getattr(A, name)(A.embed(x)))
    spec = torch.stack([A.readout(s) for s in accumulate(ticks(A, getattr(A, side)(x)), A.combine)], dim=-1)
    assert close(override, spec)
    assert close(grad(override, x), grad(spec, x))


@pytest.mark.parametrize('A', lifted, ids=lambda a: type(a).__name__)
def test_lifted_span(A: Lifted):
    torch.manual_seed(0)
    x = torch.rand(batch, time) * 20 - 10
    windows = A.readout(A.span_meet(A.embed(x)))
    parts = ticks(A, A.embed_meet(x))
    for t in range(time):
        for u in range(time):
            reference = (A.readout(reduce(A.combine, parts[t:u + 1])) if u >= t
                         else A.bottom_value.expand(batch))
            assert close(windows[..., t, u], reference)
