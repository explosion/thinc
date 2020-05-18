from typing import Any, cast
import contextlib
from io import BytesIO
import srsly

try:
    import torch.autograd
    import torch.optim
    import torch
except ImportError:  # pragma: no cover
    pass

from ..util import torch2xp, xp2torch, get_array_module, convert_recursive
from ..backends import get_current_ops, get_array_ops
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from .shim import Shim


class PyTorchShim(Shim):
    """Interface between a PyTorch model and a Thinc Model. This container is
    *not* a Thinc Model subclass itself.
    """

    def __call__(self, inputs, is_train):
        if is_train:
            return self.begin_update(inputs)
        else:
            return self.predict(inputs), lambda a: ...

    def predict(self, inputs: ArgsKwargs) -> Any:
        """Pass inputs through to the underlying PyTorch model, and return the
        output. No conversions are performed. The PyTorch model is set into
        evaluation mode.
        """
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(*inputs.args, **inputs.kwargs)
        self._model.train()
        return outputs

    def begin_update(self, inputs: ArgsKwargs):
        """Pass the inputs through to the underlying PyTorch model, keeping
        track of which items in the input are tensors requiring gradients.
        If the model returns a single value, it is converted into a one-element tuple. Return the outputs and a callback to backpropagate.  """
        self._model.train()
        output = self._model(*inputs.args, **inputs.kwargs)

        def backprop(grads):
            torch.autograd.backward(*grads.args, **grads.kwargs)
            return convert_recursive(
                lambda x: hasattr(x, "grad"), lambda x: x.grad, inputs
            )

        return output, backprop

    def finish_update(self, optimizer: Optimizer):
        params = []
        grads = []
        shapes = []
        for name, torch_data in self._model.named_parameters():
            xp_data = cast(FloatsXd, torch2xp(torch_data.data))
            if torch_data.grad is not None:
                xp_grad = cast(FloatsXd, torch2xp(torch_data.grad))
            else:
                xp_grad = cast(FloatsXd, torch2xp(torch.zeros_like(torch_data)))
            params.append(xp_data.ravel())
            grads.append(xp_grad.ravel())
            shapes.append((xp_data.size, xp_data.shape))
        if not params:
            return
        xp = get_array_module(params[0])
        flat_params, flat_grads = optimizer(
            (self.id, "pytorch-shim"), xp.concatenate(params), xp.concatenate(grads)
        )
        start = 0
        for name, torch_data in self._model.named_parameters():
            size, shape = shapes.pop(0)
            param = flat_params[start : start + size].reshape(shape)
            torch_data.data = xp2torch(param, requires_grad=True)
            if torch_data.grad is not None:
                torch_data.grad.zero_()
            start += size

    @contextlib.contextmanager
    def use_params(self, params):
        key_prefix = f"pytorch_{self.id}_"
        state_dict = {}
        for k, v in params.items():
            if hasattr(k, "startswith") and k.startswith(key_prefix):
                state_dict[k.replace(key_prefix, "")] = xp2torch(v)
        if state_dict:
            backup = {k: v.clone() for k, v in self._model.state_dict().items()}
            self._model.load_state_dict(state_dict)
            yield
            self._model.load_state_dict(backup)
        else:
            yield

    def to_device(self, device):  # pragma: no cover
        if device == "cpu":
            self._model.cpu()
        else:
            self._model.cuda(device)

    def to_bytes(self):
        filelike = BytesIO()
        torch.save(self._model.state_dict(), filelike)
        filelike.seek(0)
        weights_bytes = filelike.getvalue()
        msg = {"config": self.cfg, "state": weights_bytes}
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data):
        ops = get_current_ops()
        msg = srsly.msgpack_loads(bytes_data)
        self.cfg = msg["config"]
        filelike = BytesIO(msg["state"])
        filelike.seek(0)
        if ops.device_type == "cpu":
            map_location = "cpu"
        else:  # pragma: no cover
            device_id = torch.cuda.current_device()
            map_location = "cuda:%d" % device_id
        self._model.load_state_dict(torch.load(filelike, map_location=map_location))
        self._model.to(map_location)
        return self
