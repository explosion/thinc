from srsly import cloudpickle as pickle
from thinc.api import FeatureExtractor


def test_pickle():
    model = FeatureExtractor([100, 200])
    bytes_data = pickle.dumps(model)
    loaded = pickle.loads(bytes_data)
    assert loaded._attrs == model._attrs
