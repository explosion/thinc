from typing import Callable, Tuple, Any
from ..model import Model
from ..shims import TensorFlowShim
from ..util import xp2tensorflow, tensorflow2xp
from ..types import Array

try:
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False


def TensorFlowWrapper(tensorflow_model: Any) -> Model:
    """Wrap a TensorFlow model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a Tensorflow optimizer and call
    optimizer.apply_gradients after each batch
    """
    assert has_tensorflow, "Tensorflow not found!"
    assert isinstance(tensorflow_model, tf.keras.models.Model), \
        "tensorflow_model must be an instance of tf.keras.models.Model"
    return Model("tensorflow", forward, shims=[TensorFlowShim(tensorflow_model)])


def forward(model: Model, X: Array, is_train: bool) -> Tuple[Array, Callable]:
    """Return the output of the wrapped TensorFlow model for the given input,
    along with a callback to handle the backward pass.
    """
    tensorflow_model = model.shims[0]
    X_tensorflow = xp2tensorflow(X, requires_grad=is_train)
    Y_tensorflow, tensorflow_backprop = tensorflow_model((X_tensorflow,), {}, is_train)
    Y = tensorflow2xp(Y_tensorflow)

    def backprop(dY):
        dY_tensorflow = xp2tensorflow(dY, requires_grad=is_train)
        dX_tensorflow = tensorflow_backprop((dY_tensorflow,), {})
        # handling multiple inputs
        if isinstance(dX_tensorflow, list):
            dX_tensorflow = [tensorflow2xp(g) for g in dX_tensorflow]
            return dX_tensorflow
        else:
            return tensorflow2xp(dX_tensorflow)

    return Y, backprop
