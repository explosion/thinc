import pytest
from mock import MagicMock
from functools import partial
from hypothesis import given, settings
import numpy
from numpy.testing import assert_allclose
from thinc.api import resizable, Linear, chain, Dropout, SGD
from thinc.layers.resizable import resize_model, resize_linear_weighted

from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape


@pytest.fixture
def model():
    output_layer = Linear(nO=None, nI=None)
    fill_defaults = {"b": 0, "W": 0}
    model = resizable(
        output_layer,
        resize_layer=partial(resize_linear_weighted, fill_defaults=fill_defaults),
    )
    return model


def test_resizable_linear_default_name(model):
    assert model.name == "resizable(linear)"


def test_resize_model(model):
    """Test that resizing the model doesn't cause an exception."""
    resize_model(model, new_nO=10)
    resize_model(model, new_nO=11)

    model.set_dim("nO", 0, force=True)
    resize_model(model, new_nO=10)

    model.set_dim("nI", 10, force=True)
    model.set_dim("nO", 0, force=True)
    resize_model(model, new_nO=10)
