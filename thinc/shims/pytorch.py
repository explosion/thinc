import contextlib
from io import BytesIO
from ..util import torch2xp, xp2torch

from .shim import Shim

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch.autograd
    import torch.optim
    import torch
except ImportError:
    has_torch = False


class PyTorchShim(Shim):
    """Interface between a PyTorch model and a Thinc Model. This container is
    *not* a Thinc Model subclass itself.
    """

    def __call__(self, args, kwargs, is_train):
        if is_train:
            return self.begin_update(args, kwargs)
        else:
            return self.predict(args, kwargs), lambda args, kwargs: (args, kwargs)

    def predict(self, args, kwargs):
        self._model.eval()
        with torch.no_grad():
            y_var = self._model(*args, **kwargs)
        self._model.train()
        return y_var, lambda d_args, d_kwargs: d_args

    def begin_update(self, args, kwargs):
        self._model.train()
        output = self._model(*args, **kwargs)

        def backprop(d_args, d_kwargs):
            torch.autograd.backward(*d_args, **d_kwargs)
            return args

        return output, backprop

    def finish_update(self, optimizer):
        if not self._optimizer:
            self._optimizer = self._create_optimizer(optimizer)
        if getattr(optimizer, "max_grad_norm", None):
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), optimizer.max_grad_norm
            )
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._update_pytorch_averages(optimizer)

    def _create_optimizer(self, sgd):
        params = self._model.parameters()
        if sgd.b1 != 0 and sgd.b2 != 0:
            optimizer = torch.optim.Adam(params, lr=sgd.alpha, betas=(sgd.b1, sgd.b2))
        elif sgd.b2 == 0:
            optimizer = torch.optim.SGD(params, lr=sgd.alpha, momentum=sgd.b1)
        else:
            raise NotImplementedError
        return optimizer

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

    def _update_pytorch_averages(self, sgd, *, init_steps=1):
        if getattr(sgd, "averages", None) is None:
            return
        # Collect parameters if we don't have them
        for name, param in self._model.state_dict().items():
            key = f"pytorch_{self.id}_{name}"
            sgd.nr_update[key] += 1
            xp_param = torch2xp(param)
            if key in sgd.averages:
                sgd.ops.update_averages(sgd.averages[key], xp_param, sgd.nr_update[key])
            else:
                sgd.averages[key] = xp_param.copy()
                sgd.nr_update[key] = init_steps

    def to_gpu(self, device_num):
        self._model.cuda(device_num)

    def to_cpu(self):
        self._model.cpu()

    def to_disk(self, path):
        torch.save(self._model.state_dict(), str(path))

    def from_disk(self, path):
        if self.ops.device == "cpu":
            map_location = "cpu"
        else:
            device_id = torch.cuda.current_device()
            map_location = "cuda:%d" % device_id
        self._model.load_state_dict(torch.load(path, map_location=map_location))
        self._model.to(map_location)
        return self

    def to_bytes(self):
        filelike = BytesIO()
        torch.save(self._model.state_dict(), filelike)
        filelike.seek(0)
        return filelike.getvalue()

    def from_bytes(self, data):
        filelike = BytesIO(data)
        filelike.seek(0)
        if self.ops.device == "cpu":
            map_location = "cpu"
        else:
            device_id = torch.cuda.current_device()
            map_location = "cuda:%d" % device_id
        self._model.load_state_dict(torch.load(filelike, map_location=map_location))
        self._model.to(map_location)
        return self


"""
class PyTorchWrapperRNN(PyTorchShim):
    # Wrap a PyTorch RNN model

    def prepare_input(self, inputs, is_update=False):
        if isinstance(inputs, tuple):
            x_data, h_0 = inputs
        else:
            x_data = inputs
            h_0 = None
        x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=is_update)
        return (x_var, h_0), {}

    def prepare_output(self, torch_outputs):
        y_var, h_n = torch_outputs
        return torch2xp(y_var), h_n

    def prepare_backward_input(self, dy_data, y_var):
        dy, _ = dy_data
        dy_var = xp2torch(dy)
        y_var, _ = y_var
        return (y_var,), {"grad_tensors": (dy_var,)}

    def prepare_backward_output(self, x_args, x_kwargs):
        x_var, _ = x_args
        return torch2xp(x_var.grad)

    def prepare_input(self, x_data, is_update=True):
        if isinstance(x_data, (list, tuple)):
            x_var = [
                torch.autograd.Variable(xp2torch(x), requires_grad=is_update)
                for x in x_data
            ]
            return tuple(x_var), {}
        else:
            x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=is_update)
            return (x_var,), {}

    def prepare_output(self, y_var):
        if isinstance(y_var, (list, tuple)):
            return tuple([torch2xp(y) for y in y_var])
        else:
            return torch2xp(y_var)

    def prepare_backward_input(self, dy_data, y_var):
        if isinstance(dy_data, (list, tuple)):
            dy_var = [xp2torch(dy) for dy in dy_data]
            return y_var, {"grad_tensors": dy_var}
        else:
            dy_var = xp2torch(dy_data)
            return (y_var,), {"grad_tensors": (dy_var,)}

    def prepare_backward_output(self, x_args, x_kwargs):
        x_var = x_args[0]
        return torch2xp(x_var.grad)
"""
