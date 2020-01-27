from typing import Callable, Tuple, Optional, Any, cast

from ..model import Model
from ..shims import PyTorchShim
from ..config import registry
from ..util import is_xp_array, is_torch_array
from ..util import xp2torch, torch2xp, convert_recursive
from ..types import Floats3d, ArgsKwargs, Padded


@registry.layers("PyTorchRNNWrapper.v1")
def PyTorchRNNWrapper(
    pytorch_model,
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
) -> Model[Padded, Padded]:
    """Wrap a PyTorch RNN model for use in Thinc."""
    if convert_inputs is None:
        convert_inputs = convert_rnn_inputs
    if convert_outputs is None:
        convert_outputs = convert_rnn_outputs
    return cast(
        Model[Padded, Padded],
        PyTorchWrapper(
            pytorch_model,
            convert_inputs=convert_inputs,
            convert_outputs=convert_outputs,
        ),
    )


@registry.layers("PyTorchWrapper.v1")
def PyTorchWrapper(
    pytorch_model,
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
) -> Model[Any, Any]:
    """Wrap a PyTorch model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch. See examples/wrap_pytorch.py

    Your PyTorch model's forward method can take arbitrary args and kwargs,
    but must return either a single tensor as output or a tuple. You may find the
    PyTorch register_forward_hook helpful if you need to adapt the output.

    The convert functions are used to map inputs and outputs to and from your
    PyTorch model. Each function should return the converted output, and a callback
    to use during the backward pass. So:

        Xtorch, get_dX = convert_inputs(X)
        Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
        Y, get_dYtorch = convert_outputs(Ytorch)

    To allow maximum flexibility, the PyTorchShim expects ArgsKwargs objects
    on the way into the forward and backward passed. The ArgsKwargs objects
    will be passed straight into the model in the forward pass, and straight
    into `torch.autograd.backward` during the backward pass.
    """
    if convert_inputs is None:
        convert_inputs = convert_pytorch_default_inputs
    if convert_outputs is None:
        convert_outputs = convert_pytorch_default_outputs
    return Model(
        "pytorch",
        forward,
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
        shims=[PyTorchShim(pytorch_model)],
    )


def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
    """Return the output of the wrapped PyTorch model for the given input,
    along with a callback to handle the backward pass.
    """
    convert_inputs = model.attrs["convert_inputs"]
    convert_outputs = model.attrs["convert_outputs"]

    Xtorch, get_dX = convert_inputs(model, X, is_train)
    Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
    Y, get_dYtorch = convert_outputs(model, (X, Ytorch), is_train)

    def backprop(dY: Any) -> Any:
        dYtorch = get_dYtorch(dY)
        dXtorch = torch_backprop(dYtorch)
        dX = get_dX(dXtorch)
        return dX

    return Y, backprop


# Default conversion functions


def convert_pytorch_default_inputs(
    model: Model, X: Any, is_train: bool
) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Any]]:
    xp2torch_ = lambda x: xp2torch(x, requires_grad=is_train)
    converted = convert_recursive(is_xp_array, xp2torch_, X)
    if isinstance(converted, ArgsKwargs):

        def reverse_conversion(dXtorch):
            return convert_recursive(is_torch_array, torch2xp, dXtorch)

        return converted, reverse_conversion
    elif isinstance(converted, dict):

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
            return dX.kwargs

        return ArgsKwargs(args=tuple(), kwargs=converted), reverse_conversion
    elif isinstance(converted, (tuple, list)):

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
            return dX.args

        return ArgsKwargs(args=tuple(converted), kwargs={}), reverse_conversion
    else:

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_torch_array, torch2xp, dXtorch)
            return dX.args[0]

        return ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion


def convert_pytorch_default_outputs(model: Model, X_Ytorch: Any, is_train: bool):
    X, Ytorch = X_Ytorch
    Y = convert_recursive(is_torch_array, torch2xp, Ytorch)

    def reverse_conversion(dY: Any) -> ArgsKwargs:
        dYtorch = convert_recursive(is_xp_array, xp2torch, dY)
        return ArgsKwargs(args=((Ytorch,),), kwargs={"grad_tensors": dYtorch})

    return Y, reverse_conversion


# BiLSTM conversion functions


def convert_rnn_inputs(model: Model, Xp: Padded, is_train: bool):
    size_at_t = Xp.size_at_t
    lengths = Xp.lengths
    indices = Xp.indices

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Padded:
        dX = torch2xp(d_inputs.args[0])
        return Padded(dX, size_at_t, lengths, indices)  # type: ignore

    output = ArgsKwargs(args=(xp2torch(Xp.data, requires_grad=True), None), kwargs={})
    return output, convert_from_torch_backward


def convert_rnn_outputs(model: Model, inputs_outputs: Tuple, is_train):
    Xp, (Ytorch, _) = inputs_outputs

    def convert_for_torch_backward(dYp: Padded) -> ArgsKwargs:
        dYtorch = xp2torch(dYp.data, requires_grad=True)
        return ArgsKwargs(args=(Ytorch,), kwargs={"grad_tensors": dYtorch})

    Y = cast(Floats3d, torch2xp(Ytorch))
    Yp = Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices)
    return Yp, convert_for_torch_backward
