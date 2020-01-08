from thinc.api import xp2tensorflow, tensorflow2xp
import numpy
import pytest

try:
    import tensorflow as tf

    has_tensorflow = True
except ImportError:
    has_tensorflow = False


@pytest.mark.skipif(not has_tensorflow, reason="needs TensorFlow")
def test_roundtrip_conversion():
    xp_tensor = numpy.zeros((2, 3), dtype="f")
    tf_tensor = xp2tensorflow(xp_tensor)
    assert isinstance(tf_tensor, tf.Tensor)
    new_xp_tensor = tensorflow2xp(tf_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)
