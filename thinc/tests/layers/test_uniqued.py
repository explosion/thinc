import pytest
import numpy
from thinc.layers import uniqued, Embed
from numpy.testing import assert_allclose
from hypothesis import given
from hypothesis.strategies import integers, lists, composite

ROWS = 10

@composite
def lists_of_integers(draw, columns=2, lo=0, hi=ROWS):
    int_list = draw(lists(integers(min_value=lo, max_value=hi)))
    # Trim so we're of length divisible by columns.
    int_list = int_list[len(int_list) % columns:]
    array = numpy.array(int_list, dtype="uint64")
    return array.reshape((-1, columns))


@pytest.fixture
def model(nO=128):
    return Embed(nO, ROWS, column=0)


@given(X=lists_of_integers(lo=0, hi=ROWS))
def test_uniqued_doesnt_change_result(model, X):
    if X.size == 0:
        return
    umodel = uniqued(model, column=1)
    Y, bp_Y = model(X, is_train=True)
    Yu, bp_Yu = umodel(X, is_train=True)
    assert_allclose(Y, Yu)
    dX = bp_Y(Y)
    dXu = bp_Yu(Yu)
    assert_allclose(dX, dXu)
    # Check that different inputs do give different results
    Z, bp_Z = model(X+1, is_train=True)
    with pytest.raises(AssertionError):
        assert_allclose(Y, Z)
