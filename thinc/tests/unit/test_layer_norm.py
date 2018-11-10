from __future__ import unicode_literals
import pytest
from ...neural._classes.layernorm import LayerNorm
from ...neural._classes.affine import Affine

@pytest.fixture
def child():
    return Affine(5, 8)

@pytest.fixture
def model(child):
    return LayerNorm(child)
 
def test_LayerNorm_init(child):
    model = LayerNorm(child)

def test_LayerNorm_default_name(model):
    assert model.name == 'layernorm'


