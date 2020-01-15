from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np

from ..config import registry
from ..model import Model
from ..shims import TensorFlowShim
from ..types import ArgsKwargs, Array
from ..util import (
    assert_tensorflow_installed,
    convert_recursive,
    is_tensorflow_array,
    is_xp_array,
    tensorflow2xp,
    xp2tensorflow,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    pass


InT = TypeVar("InT")
OutT = TypeVar("OutT")


InFunc = TypeVar("InFunc")
XType = TypeVar("XType", bound=Array)
YType = TypeVar("YType", bound=Array)


def keras_subclass(
    name: str,
    X: XType,
    Y: YType,
    input_shape: Tuple[int, ...],
    args: Optional[Dict[str, Any]] = None,
) -> Callable[[InFunc], InFunc]:
    """Decorate a custom keras subclassed model with enough information to
    serialize and deserialize it reliably in the face of the many restrictions
    on keras subclassed models.

    name (str): The unique namespace string to use to represent this model class.
    X (Any): A sample X input for performing a forward pass on the network.
    Y (Any): A sample Y input for performing a backward pass on the network.
    input_shape (Tuple[int, ...]): A set of input shapes for building the network. 
    args: Additional arguments are passed to the class constructor

    RETURNS (Callable): The decorated class.
    """

    def call_fn(clazz):
        clazz.catalogue_name = property(lambda inst: name)
        clazz.eg_shape = property(lambda inst: input_shape)
        clazz.eg_x = property(lambda inst: X)
        clazz.eg_y = property(lambda inst: Y)

        @registry.keras(name)
        def create_component(**call_kwargs):
            input_args = call_kwargs
            if args is not None:
                input_args = {**args, **call_kwargs}
            return clazz(**input_args)

        return clazz

    return call_fn


def TensorFlowWrapper(
    tensorflow_model: Any,
    build_model: bool = True,
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
    optimizer: Optional[Any] = None,
    model_class: Type[Model] = Model,
    input_shape: Optional[Tuple[int, ...]] = None,
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

    # Determine if the model is Sequential/Functional
    is_subclass = False
    try:
        tensorflow_model.to_json()
    except NotImplementedError:
        is_subclass = True

    if is_subclass:
        for prop_name in ["catalogue_name", "eg_x", "eg_y", "eg_shape"]:
            if not hasattr(tensorflow_model, prop_name):
                raise ValueError(
                    "Keras subclassed models are not whole-model serializable by "
                    "Tensorflow. To work around this, you must decorate your keras "
                    "model subclasses with the 'keras_subclass' decorator. The decorator "
                    "requires a single X/Y input of fake-data that can be used to initialize "
                    "your subclass model properly when loading the saved version."
                )
        # Attach the input shape if it's not provided
        if input_shape is None:
            input_shape = tensorflow_model.eg_shape

    # Building a TF model checks for errors like not specifying an input_shape
    # which can cause other errors in methods like from_disk and from_bytes.
    if build_model:
        tensorflow_model.build(input_shape=input_shape)
    if convert_inputs is None:
        convert_inputs = _convert_inputs
    if convert_outputs is None:
        convert_outputs = _convert_outputs
    return model_class(
        model_name,
        forward,
        shims=[TensorFlowShim(tensorflow_model, optimizer=optimizer)],
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
