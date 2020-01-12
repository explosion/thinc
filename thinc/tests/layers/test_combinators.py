import pytest
import numpy
from thinc.api import chain, clone, concatenate, noop, Linear, Model


@pytest.fixture(params=[1, 2, 9])
def nB(request):
    return request.param


@pytest.fixture(params=[1, 6])
def nI(request):
    return request.param


@pytest.fixture(params=[1, 5, 3])
def nH(request):
    return request.param


@pytest.fixture(params=[1, 2, 7, 9])
def nO(request):
    return request.param


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
    with pytest.raises(TypeError):
        chain()


def test_chain_one(model1):
    with pytest.raises(TypeError):
        chain(model1)


def test_chain_two(model1, model2):
    model = chain(model1, model2)
    assert len(model.layers) == 2


def test_chain_operator_two(model1, model2):
    with Model.define_operators({">>": chain}):
        model = model1 >> model2
        assert len(model.layers) == 2


def test_chain_three(model1, model2, model3):
    model = chain(model1, model2, model3)
    assert len(model.layers) == 3


def test_chain_operator_three(model1, model2, model3):
    with Model.define_operators({">>": chain}):
        model = model1 >> model2 >> model3
        assert len(model.layers) == 3


def test_concatenate_one(model1):
    model = concatenate(model1)
    assert isinstance(model, Model)


def test_concatenate_two(model1, model2):
    model = concatenate(model1, model2)
    assert len(model.layers) == 2


def test_concatenate_operator_two(model1, model2):
    with Model.define_operators({"|": concatenate}):
        model = model1 | model2
        assert len(model.layers) == 2


def test_concatenate_three(model1, model2, model3):
    model = concatenate(model1, model2, model3)
    assert len(model.layers) == 3


def test_concatenate_operator_three(model1, model2, model3):
    with Model.define_operators({"|": concatenate}):
        model = model1 | model2 | model3
        assert len(model.layers) == 3


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


def test_clone_noop():
    model = clone(Linear(), 0)
    assert len(model.layers) == 0
    assert model.name == "noop"


def test_noop():
    data = numpy.asarray([1, 2, 3], dtype="f")
    model = noop(Linear(), Linear())
    model.initialize(data, data)
    Y, backprop = model(data)
    assert numpy.array_equal(Y, data)
    dX = backprop(Y)
    assert numpy.array_equal(dX, data)
