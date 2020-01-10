import numpy
import pytest
from ..util import make_tempdir

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
        guesses, backprop = model(X, is_train=True)
        d_guesses = (guesses - Y) / guesses.shape[0]
        backprop(d_guesses)
        model.finish_update(optimizer)
    predicted = model.predict(X).argmax()
    assert predicted == answer


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_can_copy_model(model: Model[FloatsNd, FloatsNd]):
    copy: Model[FloatsNd, FloatsNd] = model.copy()
    assert copy is not None


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_print_summary(
    model: Model[FloatsNd, FloatsNd], X: FloatsNd
):
    # Cannot print a keras summary until shapes are known
    model.predict(X)
    summary = str(model.shims[0])
    # Summary includes the layers of our model
    assert "layer_normalization" in summary
    assert "dense" in summary
    # And counts of params
    assert "Total params" in summary
    assert "Trainable params" in summary
    assert "Non-trainable params" in summary


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_to_bytes(model: Model[FloatsNd, FloatsNd], X: FloatsNd):
    # Keras doesn't create weights until the model is called or built
    with pytest.raises(ValueError):
        model_bytes = model.to_bytes()
    # After predicting, the model is built
    model.predict(X)
    # And can be serialized
    model_bytes = model.to_bytes()
    assert model_bytes is not None


@pytest.mark.skip("keras load fails with weights mismatch")
def test_tensorflow_wrapper_to_from_disk(
    model: Model[FloatsNd, FloatsNd], X: FloatsNd, Y: FloatsNd, answer: int
):
    optimizer = Adam()
    guesses, backprop = model(X, is_train=True)
    backprop((guesses - Y) / guesses.shape[0])
    model.finish_update(optimizer)
    model.predict(X)
    with make_tempdir() as tmp_path:
        model_file = tmp_path / "model.h5"
        model.to_disk(model_file)
        another_model = model.from_disk(model_file)
        assert another_model is not None


@pytest.mark.skip("keras load fails with weights mismatch")
def test_tensorflow_wrapper_from_bytes(model: Model[FloatsNd, FloatsNd], X: FloatsNd):
    model.predict(X)
    model_bytes = model.to_bytes()
    another_model = model.from_bytes(model_bytes)
    assert another_model is not None


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_use_params(
    model: Model[FloatsNd, FloatsNd], X: FloatsNd, Y: FloatsNd, answer: int
):
    optimizer = Adam()
    with model.use_params(optimizer.averages):
        assert model.predict(X).argmax() is not None
    for i in range(10):
        guesses, backprop = model.begin_update(X)
        d_guesses = (guesses - Y) / guesses.shape[0]
        backprop(d_guesses)
        model.finish_update(optimizer)
    with model.use_params(optimizer.averages):
        predicted = model.predict(X).argmax()
    assert predicted == answer


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_to_cpu(model: Model[FloatsNd, FloatsNd], X: FloatsNd):
    model.to_cpu()


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
def test_tensorflow_wrapper_to_gpu(model: Model[FloatsNd, FloatsNd], X: FloatsNd):
    # Raises while failing to import cupy
    with pytest.raises(ModuleNotFoundError):
        model.to_gpu(0)
