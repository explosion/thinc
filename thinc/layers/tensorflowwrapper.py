from typing import Any, Callable, Tuple, TypeVar

from ..model import Model
from ..shims import TensorFlowShim
from ..util import xp2tensorflow, tensorflow2xp, assert_tensorflow_installed
from ..util import is_tensorflow_array, convert_recursive, is_xp_array
from ..types import ArgsKwargs

try:
    import tensorflow as tf
except ImportError:
    pass

InT = TypeVar("InT")
OutT = TypeVar("OutT")


def TensorFlowWrapper(
    tensorflow_model: Any, build_model: bool = True
) -> Model[InT, OutT]:
    """Wrap a TensorFlow model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a TensorFlow optimizer and call
    optimizer.apply_gradients after each batch.
    """
    assert_tensorflow_installed()
    if not isinstance(tensorflow_model, tf.keras.models.Model):
        err = f"Expected tf.keras.models.Model, got: {type(tensorflow_model)}"
        raise ValueError(err)
    # Building a TF model checks for errors like not specifying an input_shape
    # which can cause other errors in methods like from_disk and from_bytes.
    if build_model:
        tensorflow_model.build()
    return Model("tensorflow", forward, shims=[TensorFlowShim(tensorflow_model)])


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    """Return the output of the wrapped TensorFlow model for the given input,
    along with a callback to handle the backward pass.
    """
    tensorflow_model = model.shims[0]
    X_tensorflow, get_dX = _convert_inputs(model, X, is_train)
    if is_train:
        Y_tensorflow, tensorflow_backprop = tensorflow_model(X_tensorflow, is_train)
    else:
        Y_tensorflow = tensorflow_model(X_tensorflow, is_train)
    Y, get_dY_tensorflow = _convert_outputs(model, Y_tensorflow, is_train)

    def backprop(dY: OutT) -> InT:
        dY_tensorflow = get_dY_tensorflow(dY)
        dX_tensorflow = tensorflow_backprop(dY_tensorflow)
        return get_dX(dX_tensorflow)

    return Y, backprop


# Default conversion functions
# These are pretty much the same as the PyTorch one, but I think we should
# leave the duplication -- I think the abstraction could get pretty messy,
# and then may need to be undone, as there can always be different specifics.


def _convert_inputs(model, X, is_train):
    xp2tensorflow_ = lambda x: xp2tensorflow(x, requires_grad=is_train)
    converted = convert_recursive(is_xp_array, xp2tensorflow_, X)
    if isinstance(converted, ArgsKwargs):

        def reverse_conversion(dXtf):
            return convert_recursive(is_tensorflow_array, tensorflow2xp, dXtf)

        return converted, reverse_conversion
    elif isinstance(converted, dict):

        def reverse_conversion(dXtf):
            dX = convert_recursive(is_tensorflow_array, tensorflow2xp, dXtf)
            return dX.kwargs

        return ArgsKwargs(args=tuple(), kwargs=converted), reverse_conversion
    elif isinstance(converted, (tuple, list)):

        def reverse_conversion(dXtf):
            dX = convert_recursive(is_tensorflow_array, tensorflow2xp, dXtf)
            return dX.args

        return ArgsKwargs(args=converted, kwargs={}), reverse_conversion
    else:

        def reverse_conversion(dXtf):
            dX = convert_recursive(is_tensorflow_array, tensorflow2xp, dXtf)
            return dX.args[0]

        return ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion


def _convert_outputs(model, Ytf, is_train):
    Y = convert_recursive(is_tensorflow_array, tensorflow2xp, Ytf)

    def reverse_conversion(dY):
        return convert_recursive(is_xp_array, xp2tensorflow, dY)

    return Y, reverse_conversion
