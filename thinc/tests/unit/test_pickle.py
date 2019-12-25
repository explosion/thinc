import pickle
from thinc.linear.linear import LinearModel


def test_pickle_linear_model():
    model = LinearModel(10)
    pickle.loads(pickle.dumps(model))
