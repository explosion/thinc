'''Test feature extracter.'''
from __future__ import unicode_literals

from ...api import FeatureExtracter
import pickle

def test_pickle():
    model = FeatureExtracter([100, 200])
    bytes_data = pickle.dumps(model)
    loaded = pickle.loads(bytes_data)
    assert loaded.attrs == model.attrs
