import numpy
import pytest


@pytest.mark.xfail
def test_init():
    from thinc.layers.sparse_linear import LinearModel
    model = LinearModel(3)
    keys = numpy.ones((5,), dtype="uint64")
    values = numpy.ones((5,), dtype="f")
    lengths = numpy.zeros((2,), dtype=numpy.int_)
    lengths[0] = 3
    lengths[1] = 2
    scores, backprop = model.begin_update((keys, values, lengths))
    assert scores.shape == (2, 3)
    d_feats = backprop(scores)
    assert d_feats is None
