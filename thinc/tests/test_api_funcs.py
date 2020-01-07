import pytest
import numpy
from thinc.layers import chain, clone, Linear
from thinc.model import Model


@pytest.fixture
def model1(nH, nI):
    return Linear(nH, nI)


@pytest.fixture
def model2(nO, nH):
    return Linear(nO, nH)


@pytest.fixture
def model3(nO):
    return Linear(nO, nO)


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


def test_clone_changes_predictions(nH, nI):
    model1 = Linear(nH)
    model = clone(model1, 10)
    ones = numpy.ones((10, nI), dtype="f")
    model.initialize(X=ones)
    output_from_cloned = model.predict(ones)
    output_from_orig = model1.predict(ones)
    assert output_from_cloned.sum() != output_from_orig.sum()


def test_clone_gives_distinct_ids(nH, nI):
    model = clone(Linear(nH), 5)
    assert len(model.layers) == 5
    seen_ids = set()
    for node in model.walk():
        assert node.id not in seen_ids
        seen_ids.add(node.id)
    assert len(seen_ids) == 6
