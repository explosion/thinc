import pytest

from ..api import chain
from ..neural._classes.affine import Affine


@pytest.fixture
def model1(nH, nI):
    return Affine(nH, nI)


@pytest.fixture
def model2(nO, nH):
    return Affine(nO, nH)


@pytest.fixture
def model3(nO):
    return Affine(nO, nO)


def test_chain_one(model1, model2):
    model = chain(model1, model2)
    assert len(model._layers) == 2

def test_chain_right_branch(model1, model2, model3):
    merge1 = chain(model1, model2)
    merge2 = chain(merge1, model3)
    assert len(merge2._layers) == 2


def test_chain_predict(model1, model2):
    pass


def test_chain_dimension_mismatch(model1, model2):
    pass


def test_chain_begin_update(model1, model2):
    pass


def test_chain_learns(model1, model2):
    pass
