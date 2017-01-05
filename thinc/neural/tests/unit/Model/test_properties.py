import pytest

from ....base import Model
from ....ops import NumpyOps


@pytest.fixture
def model():
    model = Model(ops=NumpyOps())
    return model

def test_can_get_describe_params(model):
    describe_params = list(model.describe_params)

def test_cant_set_describe_params(model):
    with pytest.raises(AttributeError):
        model.describe_params = 'hi'

def test_can_get_shape(model):
    shape = model.shape

def test_can_set_shape(model):
    model.shape = 'hi'

def test_can_get_input_shape(model):
    input_shape = model.input_shape

def test_can_set_input_shape(model):
    model.input_shape = (10,)


def test_can_get_output_shape(model):
    output_shape = model.output_shape

def test_can_set_output_shape(model):
    model.output_shape = (5,)
