from ..neural._classes.model import Model

try:
    import torch.autograd
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
        x_var = torch.autograd.Variable(torch.Tensor(x_data),
                                        requires_grad=True)
        # Make prediction
        y_var = self._model(x_var)
        def backward_pytorch(dy_data, sgd=None):
            dy_var = torch.autograd.Variable(torch.Tensor(dy_data))
            torch.autograd.backward((y_var,), grad_variables=(dy_var,))
            dX = self.ops.asarray(x_var.grad.data)
        return self.ops.asarray(y_var.data), backward
