import pytest
import numpy
from numpy.testing import assert_allclose
from thinc.layers.chain import chain
from thinc.layers.affine import Affine
from thinc.model import Model


@pytest.fixture
def model1(nH, nI):
    return Affine(nH, nI)


@pytest.fixture
def model2(nO, nH):
    return Affine(nO, nH)


@pytest.fixture
def model3(nO):
    return Affine(nO, nO)


def test_chain_zero():
    model = chain()
    assert isinstance(model, Model)


def test_chain_one(model1):
    model = chain(model1)
    assert isinstance(model, Model)


def test_chain_two(model1, model2):
    model = chain(model1, model2)
    assert len(model.layers) == 2


def test_chain_right_branch(model1, model2, model3):
    merge1 = chain(model1, model2)
    merge2 = chain(merge1, model3)
    assert len(merge2.layers) == 3


@pytest.mark.xfail
def test_clone(model1, nI):
    ones = numpy.ones((10, nI), dtype="f")
    model1.nI = None
    model = clone(model1, 10)
    model.begin_training(ones)
    output_from_cloned = model.predict(ones)
    output_from_orig = model1.predict(ones)
    assert output_from_cloned.sum() != output_from_orig.sum()
