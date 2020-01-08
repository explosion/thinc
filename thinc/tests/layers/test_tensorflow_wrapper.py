import numpy
import pytest

from thinc.api import TensorFlowWrapper, tensorflow2xp, xp2tensorflow
from thinc.backends import Ops, get_current_ops
from thinc.model import Model
from thinc.optimizers import Adam
from thinc.types import FloatsNd
from thinc.util import has_tensorflow, to_categorical


@pytest.fixture
def n_hidden() -> int:
    return 12


@pytest.fixture
def n_classes() -> int:
    return 10


@pytest.fixture
def answer() -> int:
    return 1


@pytest.fixture
def X() -> FloatsNd:
    ops: Ops = get_current_ops()
    return ops.alloc(shape=(1, 784))


@pytest.fixture
def Y(answer: int, n_classes: int) -> FloatsNd:
    ops: Ops = get_current_ops()
    return to_categorical(ops.asarray([answer]), n_classes=n_classes)


@pytest.fixture
def model(n_hidden: int) -> Model[FloatsNd, FloatsNd]:
    import tensorflow as tf

    tf_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(n_hidden, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(n_hidden, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return TensorFlowWrapper(tf_model)


@pytest.mark.skipif(not has_tensorflow, reason="needs TensorFlow")
def test_roundtrip_conversion():
    import tensorflow as tf

    xp_tensor = numpy.zeros((2, 3), dtype="f")
    tf_tensor = xp2tensorflow(xp_tensor)
    assert isinstance(tf_tensor, tf.Tensor)
    new_xp_tensor = tensorflow2xp(tf_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_predict(model: Model[FloatsNd, FloatsNd], X: FloatsNd):
    model.predict(X)


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_train_overfits(
    model: Model[FloatsNd, FloatsNd], X: FloatsNd, Y: FloatsNd, answer: int
):
    optimizer = Adam()
    for i in range(100):
        guesses, backprop = model.begin_update(X)
        d_guesses = (guesses - Y) / guesses.shape[0]
        backprop(d_guesses)
        model.finish_update(optimizer)
    predicted = model.predict(X).argmax()

    assert predicted == answer
