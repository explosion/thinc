from typing import Callable, Tuple, Optional, Any

from ..model import Model
from ..shims import PyTorchShim
from ..config import registry
from ..util import is_xp_array, is_torch_array
from ..util import xp2torch, torch2xp, convert_recursive
from ..types import Array, ArgsKwargs


InT = Array
OutT = Array


@registry.layers("PyTorchWrapper.v0")
def PyTorchWrapper(
    pytorch_model,
    convert_inputs=None,
    convert_outputs=None,
    gradient_map: Optional[Tuple[int, ...]] = None,
) -> Model[InT, OutT]:
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
    return Model(
        "pytorch",
        forward,
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
        shims=[PyTorchShim(pytorch_model)],
    )


def forward(model: Model, X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    """Return the output of the wrapped PyTorch model for the given input,
    along with a callback to handle the backward pass.
    """
    convert_inputs = model.get_attr("convert_inputs") or _convert_inputs
    convert_outputs = model.get_attr("convert_outputs") or _convert_outputs

    Xtorch, get_dX = convert_inputs(model, X, is_train)
    Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
    Y, get_dYtorch = convert_outputs(model, Ytorch, is_train)

    def backprop(dY: OutT) -> InT:
        dYtorch = get_dYtorch(dY)
        dXtorch = torch_backprop(dYtorch)
        dX = get_dX(dXtorch)
        return dX

    return Y, backprop


# Default conversion functions


def _convert_inputs(model: Model, X: Any, is_train: bool) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Any]]:
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


def _convert_outputs(model: Model, Ytorch: Any, is_train: bool):
    Y = convert_recursive(is_torch_array, torch2xp, Ytorch)

    def reverse_conversion(dY: Any) -> ArgsKwargs:
        dYtorch = convert_recursive(is_xp_array, xp2torch, dY)
        return ArgsKwargs(args=((Ytorch,),), kwargs={"grad_tensors": dYtorch})

    return Y, reverse_conversion
