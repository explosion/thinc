import pytest
from srsly import cloudpickle as pickle


@pytest.mark.xfail
def test_pickle():
    from thinc.layers.featureextractor import FeatureExtractor
    model = FeatureExtracter([100, 200])
    bytes_data = pickle.dumps(model)
    loaded = pickle.loads(bytes_data)
    assert loaded.attrs == model.attrs
