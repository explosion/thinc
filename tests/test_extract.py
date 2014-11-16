import pytest

from thinc.features import SumFeat
from thinc.features import Extractor
from thinc.instance import Instance

import numpy

n0w = 0
n0p = 1
p1w = 2
p1p = 3

@pytest.fixture
def atoms():
    return numpy.array([5, 20, 10, 7], dtype=numpy.int32)

@pytest.fixture
def ex1():
    templates = [(n0w,), (n0p,), (p1w,), (p1p,)]
    return Extractor(templates, [SumFeat for _ in templates])


@pytest.fixture
def ins1(atoms, ex1):
    return Instance(atoms.shape[0], ex1.n, 5)

@pytest.fixture
def ex2():
    templates = [(n0w, p1w), (n0p, p1p), (p1w,), (p1p, n0w), (n0w, n0p)]
    return Extractor(templates, [ConjFeat for _ in templates])


def test_len1_init(ex1):
    assert ex1.n == 5


def test_len1_extract(atoms, ex1, ins1):
    assert ex1.n == 5
    ins1.extract(atoms, ex1)
    feats = ins1.feats
    assert len(feats) == ex1.n
    assert feats[0] == 1
    assert feats[1] == atoms[n0w] == 5
    assert feats[2] == atoms[n0p] == 20
    assert feats[3] == atoms[p1w] == 10
    assert feats[4] == atoms[p1p] == 7
