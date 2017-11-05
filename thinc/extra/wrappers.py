from ..compat import BytesIO
from ..neural._classes.model import Model

try:
    import torch.autograd
    import torch
except ImportError:
    pass


class PytorchWrapper(Model):
    '''Wrap a PyTorch model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch --- see examples/wrap_pytorch.py
    '''
    def __init__(self, model):
        Model.__init__(self)
        self._model = model

    def begin_update(self, x_data, drop=0.):
        '''Return the output of the wrapped PyTorch model for the given input,
        along with a callback to handle the backward pass.
        '''
        x_var = torch.autograd.Variable(torch.Tensor(x_data),
                                        requires_grad=True)
        # Make prediction
        y_var = self._model(x_var)
        def backward_pytorch(dy_data, sgd=None):
            dy_var = torch.autograd.Variable(torch.Tensor(dy_data))
            torch.autograd.backward((y_var,), grad_variables=(dy_var,))
            dX = self.ops.asarray(x_var.grad.data)
            if sgd is not None:
                optimizer.step()
            return dX
        return self.ops.asarray(y_var.data), backward

    def to_disk(self, path):
        # TODO: Untested
        torch.save(self._model.state_dict(), str(path))

    def from_disk(self, path):
        # TODO: Untested
        self._model.load_state_dict(torch.load(path))

    def to_bytes(self):
        # TODO: Untested
        filelike = BytesIO()
        torch.save(self._model.state_dict(), filelike)
        return filelike.read()

    def from_bytes(self, data):
        # TODO: Untested
        filelike = BytesIO(data)
        self._model.load_state_dict(torch.load(filelike))

    def to_gpu(self, device_num):
        # TODO: Implement
        raise NotImplementedError

    def to_cpu(self):
        # TODO: Implement
        raise NotImplementedError

    def resize_output(self):
        # TODO: Required for spaCy add label
        raise NotImplementedError

    def resize_input(self):
        # TODO: Not required yet, but should be useful
        raise NotImplementedError

    @contextlib.contextmanager
    def use_params(self, params): # pragma: no cover
        # TODO: Implement
        raise NotImplementedError

