from typing import Callable, Tuple, Any

from ..model import Model
from ..shims import TensorFlowShim
from ..util import xp2tensorflow, tensorflow2xp, assert_tensorflow_is_installed
from ..types import Array

try:
    import tensorflow as tf
except ImportError:
    pass


InT = Array
OutT = Array


def TensorFlowWrapper(tensorflow_model: Any) -> Model[InT, OutT]:
    """Wrap a TensorFlow model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a TensorFlow optimizer and call
    optimizer.apply_gradients after each batch.
    """
    assert_tensorflow_is_installed()
    if not isinstance(tensorflow_model, tf.keras.models.Model):
        err = f"Expected tf.keras.models.Model, got: {type(tensorflow_model)}"
        raise ValueError(err)
    return Model("tensorflow", forward, shims=[TensorFlowShim(tensorflow_model)])


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    """Return the output of the wrapped TensorFlow model for the given input,
    along with a callback to handle the backward pass.
    """
    tensorflow_model = model.shims[0]
    X_tensorflow = xp2tensorflow(X, requires_grad=is_train)
    Y_tensorflow, tensorflow_backprop = tensorflow_model((X_tensorflow,), {}, is_train)
    Y = tensorflow2xp(Y_tensorflow)

    def backprop(dY: OutT) -> InT:
        dY_tensorflow = xp2tensorflow(dY, requires_grad=is_train)
        dX_tensorflow = tensorflow_backprop((dY_tensorflow,), {})
        return tensorflow2xp(dX_tensorflow)

    return Y, backprop
