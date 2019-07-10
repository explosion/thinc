"""Test that various classes can be pickled and unpickled."""

import pickle
from ...linear.linear import LinearModel


def test_pickle_linear_model():
    model = LinearModel(10)
    model2 = pickle.loads(pickle.dumps(model))
