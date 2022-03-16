from typing import Any, Optional, cast
import contextlib
from io import BytesIO
import itertools
import srsly

try:
    import torch.autograd
    from torch.cuda import amp
    import torch.optim
    import torch
except ImportError:  # pragma: no cover
    pass

from ..util import torch2xp, xp2torch, convert_recursive, iterate_recursive
from ..util import has_torch_amp
from ..backends import get_current_ops, context_pools, CupyOps
from ..backends import set_gpu_allocator
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from .pytorch_grad_scaler import PyTorchGradScaler
from .shim import Shim


class PyTorchShim(Shim):
    """Interface between a PyTorch model and a Thinc Model. This container is
    *not* a Thinc Model subclass itself.

    mixed_precision:
        Enable mixed-precision. This changes whitelisted ops to run
        in half precision for better performance and lower memory use.
    grad_scaler:
        The gradient scaler to use for mixed-precision training. If this
        argument is set to "None" and mixed precision is enabled, a gradient
        scaler with the default configuration is used.
    """

    def __init__(
        self,
        model: Any,
        config=None,
        optimizer: Any = None,
        mixed_precision: bool = False,
        grad_scaler: Optional[PyTorchGradScaler] = None,
    ):
        if mixed_precision and not has_torch_amp:
            raise ValueError(
                "Mixed-precision training is not supported, requires capable GPU and torch>=1.9.0"
            )

        super().__init__(model, config, optimizer)

        if grad_scaler is None:
            grad_scaler = PyTorchGradScaler(mixed_precision)

        self._grad_scaler = grad_scaler

        self._mixed_precision = mixed_precision

        if CupyOps.xp is not None and isinstance(get_current_ops(), CupyOps):
            pools = context_pools.get()
            if "pytorch" not in pools:
                from cupy import get_default_memory_pool

                set_gpu_allocator("pytorch")
                get_default_memory_pool().free_all_blocks()

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
            with amp.autocast(self._mixed_precision):
                outputs = self._model(*inputs.args, **inputs.kwargs)
        self._model.train()
        return outputs

    def begin_update(self, inputs: ArgsKwargs):
        """Pass the inputs through to the underlying PyTorch model, keeping
        track of which items in the input are tensors requiring gradients.
        If the model returns a single value, it is converted into a one-element tuple.
        Return the outputs and a callback to backpropagate.
        """
        self._model.train()

        # Note: mixed-precision autocast must not be applied to backprop.
        with amp.autocast(self._mixed_precision):
            output = self._model(*inputs.args, **inputs.kwargs)

        def backprop(grads):
            # Normally, gradient scaling is applied to the loss of a model. However,
            # since regular thinc layers do not use mixed-precision, we perform scaling
            # locally in this shim. Scaling the loss by a factor, scales the gradients
            # by the same factor (see the chain rule). Therefore, we scale the gradients
            # backprop'ed through the succeeding layer to get the same effect as loss
            # scaling.
            grads.kwargs["grad_tensors"] = self._grad_scaler.scale(
                grads.kwargs["grad_tensors"], inplace=True
            )

            torch.autograd.backward(*grads.args, **grads.kwargs)

            # Unscale weights and check for overflows during backprop.
            grad_tensors = []
            for torch_data in itertools.chain(
                self._model.parameters(),
                iterate_recursive(lambda x: hasattr(x, "grad"), inputs),
            ):
                if torch_data.grad is not None:
                    grad_tensors.append(torch_data.grad)
            found_inf = self._grad_scaler.unscale(grad_tensors)

            # If there was an over/underflow, return zeroed-out gradients.
            if found_inf:
                grad_get = lambda x: x.grad.zero_() if x.grad is not None else x.grad
            else:
                grad_get = lambda x: x.grad

            return convert_recursive(lambda x: hasattr(x, "grad"), grad_get, inputs)

        return output, backprop

    def finish_update(self, optimizer: Optimizer):
        for name, torch_data in self._model.named_parameters():
            if torch_data.grad is not None:
                if (
                    not self._grad_scaler.found_inf
                ):  # Skip weight update if any gradient overflowed.
                    param, grad = optimizer(
                        (self.id, name),
                        cast(FloatsXd, torch2xp(torch_data.data)),
                        cast(FloatsXd, torch2xp(torch_data.grad)),
                    )
                    torch_data.data = xp2torch(param, requires_grad=True)
                torch_data.grad.zero_()

        self._grad_scaler.update()

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

    def to_device(self, device_type: str, device_id: int):  # pragma: no cover
        if device_type == "cpu":
            self._model.cpu()
        elif device_type == "gpu":
            self._model.cuda(device_id)
        else:
            msg = f"Invalid device_type: {device_type}. Try 'cpu' or 'gpu'"
            raise ValueError(msg)

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
        self._grad_scaler.to_(map_location)
        return self
