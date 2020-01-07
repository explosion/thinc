from typing import Callable, Tuple, Any

from ..model import Model
from ..shims import PyTorchShim
from ..config import registry
from ..util import xp2torch, torch2xp
from ..types import Array


InT = Array
OutT = Array


@registry.layers("pytorch_wrapper.v0")
def PyTorchWrapper(pytorch_model: Any) -> Model:
    """Wrap a PyTorch model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch --- see examples/wrap_pytorch.py
    """
    return Model("pytorch", forward, shims=[PyTorchShim(pytorch_model)])


def forward(model: Model, X: InT, is_train: bool) -> Tuple[OutT, Callable]:
    """Return the output of the wrapped PyTorch model for the given input,
    along with a callback to handle the backward pass.
    """
    pytorch_model = model.shims[0]
    X_torch = xp2torch(X, requires_grad=is_train)
    Y_torch, torch_backprop = pytorch_model((X_torch,), {}, is_train)
    Y = torch2xp(Y_torch)

    def backprop(dY: OutT) -> InT:
        dY_torch = xp2torch(dY, requires_grad=is_train)
        dX_torch = torch_backprop((Y_torch,), {"grad_tensors": dY_torch})
        return torch2xp(dX_torch[0].grad)

    return Y, backprop
