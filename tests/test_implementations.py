from __future__ import annotations

import pytest
import torch

from telos.algebras import (
    Boolean, Goedel, Lukasiewicz, Product, Robustness, Frank,
    Hamacher, Yager, SchweizerSklar, AczelAlsina, Dombi, SugenoWeber, LSE,
    KleeneDienes,
)
from telos.algebras.base import scan, fold, span


algebras = [
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

checks = [
    ('running_meet', lambda A: scan(A.meet)),
    ('running_join', lambda A: scan(A.join)),
    ('forall',       lambda A: fold(A.meet, A.top)),
    ('exists',       lambda A: fold(A.join, A.bottom)),
    ('span_meet',    lambda A: span(scan(A.meet), A.top, A.bottom)),
    ('span_join',    lambda A: span(scan(A.join), A.bottom, A.bottom)),
]


@pytest.fixture(scope='module')
def samples() -> dict[torch.dtype, torch.Tensor]:
    torch.manual_seed(0)
    return {
        torch.float32: torch.rand(batch, time),
        torch.bool: torch.rand(batch, time) > 0.5,
    }


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a, b) if a.dtype == torch.bool else torch.allclose(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('A', algebras, ids=lambda a: type(a).__name__)
@pytest.mark.parametrize('name, reference', checks, ids=[c[0] for c in checks])
def test_implementation(A, samples, name, reference):
    x = samples[A.dtype]
    assert close(getattr(A, name)(x), reference(A)(x))
