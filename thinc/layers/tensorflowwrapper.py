from typing import Any, Callable, Optional, Tuple, Type, TypeVar

from ..model import Model
from ..shims import TensorFlowShim
from ..util import xp2tensorflow, tensorflow2xp, assert_tensorflow_installed
from ..util import is_tensorflow_array, convert_recursive, is_xp_array
from ..types import ArgsKwargs

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    pass


InT = TypeVar("InT")
OutT = TypeVar("OutT")


def TensorFlowWrapper(
    tensorflow_model: Any,
    build_model: bool = True,
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
    model_class: Type[Model] = Model,
    model_name: str = "tensorflow",
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
    if convert_inputs is None:
        convert_inputs = _convert_inputs
    if convert_outputs is None:
        convert_outputs = _convert_outputs
    return model_class(
        model_name,
        forward,
        shims=[TensorFlowShim(tensorflow_model)],
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
    )


def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    """Return the output of the wrapped TensorFlow model for the given input,
    along with a callback to handle the backward pass.
    """
    convert_inputs = model.get_attr("convert_inputs")
    convert_outputs = model.get_attr("convert_outputs")
    tensorflow_model = model.shims[0]
    X_tensorflow, get_dX = convert_inputs(model, X, is_train)
    if is_train:
        Y_tensorflow, tensorflow_backprop = tensorflow_model(X_tensorflow, is_train)
    else:
        Y_tensorflow = tensorflow_model(X_tensorflow, is_train)
    Y, get_dY_tensorflow = convert_outputs(model, Y_tensorflow, is_train)

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
