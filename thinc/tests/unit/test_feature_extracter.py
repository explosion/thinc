from srsly import cloudpickle as pickle

from ...neural._classes.feature_extracter import FeatureExtracter


def test_pickle():
    model = FeatureExtracter([100, 200])
    bytes_data = pickle.dumps(model)
    loaded = pickle.loads(bytes_data)
    assert loaded.attrs == model.attrs
