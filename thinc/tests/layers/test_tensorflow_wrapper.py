import numpy
import pytest
from ..util import make_tempdir

from thinc.api import TensorFlowWrapper, tensorflow2xp, xp2tensorflow
from thinc.backends import Ops, get_current_ops
from thinc.model import Model
from thinc.layers import Linear
from thinc.optimizers import Adam
from thinc.types import FloatsNd
from thinc.util import has_tensorflow, to_categorical


@pytest.fixture
def n_hidden() -> int:
    return 12


@pytest.fixture
def input_size() -> int:
    return 784


@pytest.fixture
def n_classes() -> int:
    return 10


@pytest.fixture
def answer() -> int:
    return 1


@pytest.fixture
def X(input_size: int) -> FloatsNd:
    ops: Ops = get_current_ops()
    return ops.alloc(shape=(1, input_size))


@pytest.fixture
def Y(answer: int, n_classes: int) -> FloatsNd:
    ops: Ops = get_current_ops()
    return to_categorical(ops.asarray([answer]), n_classes=n_classes)


@pytest.fixture
def tf_model(n_hidden: int, input_size: int):
    import tensorflow as tf

    tf_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(n_hidden, input_shape=(input_size,)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(n_hidden, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return tf_model


@pytest.fixture
def model(tf_model) -> Model[FloatsNd, FloatsNd]:
    return TensorFlowWrapper(tf_model)


@pytest.mark.skipif(not has_tensorflow, reason="needs TensorFlow")
def test_tensorflow_wrapper_roundtrip_conversion():
    import tensorflow as tf

    xp_tensor = numpy.zeros((2, 3), dtype="f")
    tf_tensor = xp2tensorflow(xp_tensor)
    assert isinstance(tf_tensor, tf.Tensor)
    new_xp_tensor = tensorflow2xp(tf_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)


@pytest.mark.skipif(not has_tensorflow, reason="needs TensorFlow")
def test_tensorflow_wrapper_construction_requires_keras_model(tf_model):
    import tensorflow as tf

    with pytest.raises(ValueError):
        TensorFlowWrapper(Linear(12))
    keras_model = tf.keras.Sequential([tf.keras.layers.Dense(12, input_shape=(12,))])
    assert isinstance(TensorFlowWrapper(keras_model), Model)


@pytest.mark.skipif(not has_tensorflow, reason="needs TensorFlow")
def test_tensorflow_wrapper_built_model(
    model: Model[FloatsNd, FloatsNd], X: FloatsNd, Y: FloatsNd
):
    # built models are validated more and can perform useful operations:
    assert model.predict(X) is not None
    # Can print a keras summary
    assert str(model.shims[0]) != ""

    # They can de/serialized
    assert model.from_bytes(model.to_bytes()) is not None


@pytest.mark.skipif(not has_tensorflow, reason="needs TensorFlow")
def test_tensorflow_wrapper_unbuilt_model_hides_config_errors(
    tf_model, X: FloatsNd, Y: FloatsNd
):
    import tensorflow as tf

    # input_shape is needed to de/serialize keras models properly
    # so we throw an error as soon as we can detect that case.
    with pytest.raises(ValueError):
        TensorFlowWrapper(tf.keras.Sequential([tf.keras.layers.Dense(12)]))

    # You can override the model build at construction, but then
    # you must specify the input shape another way.
    model = TensorFlowWrapper(
        tf.keras.Sequential([tf.keras.layers.Dense(12)]), build_model=False
    )
    # Can't de/serialize without an input_shape
    with pytest.raises(ValueError):
        model.from_bytes(model.to_bytes())

    # Can't print a keras summary
    with pytest.raises(ValueError):
        str(model.shims[0])


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
    # And can be serialized
    model_bytes = model.to_bytes()
    assert model_bytes is not None


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
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


@pytest.mark.skipif(not has_tensorflow, reason="needs Tensorflow")
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
