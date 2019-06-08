# coding: utf8
from __future__ import unicode_literals

import contextlib
from ..compat import BytesIO
from ..neural._classes.model import Model

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch.autograd
    import torch.optim
    import torch
    import torch.utils.dlpack
    from torch.nn import Module as PyTorchModule
except ImportError:
    torch = None
    PyTorchModule = None


def xp2torch(xp_tensor):
    if hasattr(xp_tensor, "toDlpack"):
        return torch.utils.dlpack.from_dlpack(xp_tensor.toDlpack())
    else:
        return torch.from_numpy(xp_tensor)


def torch2xp(torch_tensor):
    if torch_tensor.is_cuda:
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    else:
        return torch_tensor.detach().numpy()


class PyTorchWrapper(Model):
    """Wrap a PyTorch model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch --- see examples/wrap_pytorch.py
    """

    def __init__(self, model, conf=None):
        Model.__init__(self)
        self._model = model
        self.conf = conf
        self._optimizer = None

    def begin_update(self, x_data, drop=0.0, **kwargs):
        """Return the output of the wrapped PyTorch model for the given input,
        along with a callback to handle the backward pass.
        """
        if self.conf is None:
            x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=True)
            y_var = self._model(x_var, **kwargs)

            def backward_pytorch(dy_data, sgd=None):
                dy_var = xp2torch(dy_data)
                torch.autograd.backward((y_var,), grad_tensors=(dy_var,))
                if sgd is not None:
                    if self._optimizer is None:
                        self._optimizer = self._create_optimizer(sgd)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                return torch2xp(x_var.grad)

            y = torch2xp(y_var)
        else:
            """
                self.grads specifies how the pytorch wrapper should behave
                when multiple inputs come into play.
                Let's say we have n inputs and m outputs
                Args:
                    i_grad: (bool iterable)
                        controls which of n inputs require grad
                    o_xp: int or None
                        determines the number of outputs
                    b_map: list or None
                        maps each dYi with one or more Yj.
                        Leave none if there is only one gradient
                    ret_x: list of int
                        a list of indexes of the inputs that will be returned
                        in the backpropagation
            """
            i_grad, o_xp, b_map, ret_x = self.conf

            """ Input numpy arrays to tensors with grad or no grad """
            x_var = [
                xp2torch(x_data[i])
                if not grad
                else torch.autograd.Variable(xp2torch(x_data[i]), requires_grad=True)
                for i, grad in enumerate(i_grad)
            ]

            y_var = self._model(x_var, **kwargs)

            """ Tensors to numpy arrays """
            if o_xp is not None:
                y = []
                for i in range(o_xp):
                    y.append(torch2xp(y_var[i]))
                y = tuple(y)
            else:
                y = torch2xp(y_var)

            def backward_pytorch(dy_data, sgd=None):
                if b_map is None:
                    # one gradient, one output
                    dy_var = xp2torch(dy_data)
                    torch.autograd.backward((y_var,), grad_tensors=(dy_var,))
                else:
                    if len(b_map) == 1:
                        # this corresponds to one gradient and multiple outputs
                        dy_var = xp2torch(dy_data)
                        grad_list = b_map[0]
                        for y_indx in grad_list:
                            torch.autograd.backward(
                                (y_var[y_indx],),
                                grad_tensors=(dy_var,),
                                retain_graph=True,
                            )
                    else:
                        # this corresponds to multiple gradients
                        vars = []
                        for grad_indx, grad_list in enumerate(b_map):
                            dy_var = xp2torch(dy_data[grad_indx])
                            for y_indx in grad_list:
                                if o_xp is not None:
                                    torch.autograd.backward(
                                        (y_var[y_indx],), grad_tensors=(dy_var,)
                                    )
                                else:
                                    torch.autograd.backward(
                                        (y_var,),
                                        grad_tensors=(dy_var,),
                                        retain_graph=True,
                                    )
                if sgd is not None:
                    if self._optimizer is None:
                        self._optimizer = self._create_optimizer(sgd)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                b_out = []
                for indx in ret_x:
                    if i_grad[indx]:
                        b_out.append(torch2xp(x_var[indx].grad))
                    else:
                        b_out.append(torch2xp(x_var[indx]))
                if len(b_out) == 0:
                    return None
                elif len(b_out) == 1:
                    return b_out[0]
                else:
                    return b_out

        return y, backward_pytorch

        return torch2xp(y_var), backward_pytorch

    def _create_optimizer(self, sgd):
        params = self._model.parameters()
        if sgd.b1 != 0 and sgd.b2 != 0:
            optimizer = torch.optim.Adam(params, lr=sgd.alpha, betas=(sgd.b1, sgd.b2))
        elif sgd.b2 == 0:
            optimizer = torch.optim.SGD(params, lr=sgd.alpha, momentum=sgd.b1)
        else:
            raise NotImplementedError
        return optimizer

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
        filelike.seek(0)
        return filelike.getvalue()

    def from_bytes(self, data):
        # TODO: Untested
        filelike = BytesIO(data)
        filelike.seek(0)
        self._model.load_state_dict(torch.load(filelike))

    def to_gpu(self, device_num):
        self._model.cuda(device_num)

    def to_cpu(self):
        self._model.cpu()

    def resize_output(self, new_dim):
        # self.weight = nn.Parameter(F.pad(self.weight, ...)) # add classes
        # self.weight = nn.Parameter(F.pad(model.weight, ...)) # add classes
        raise NotImplementedError

    def resize_input(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        if self.id in params:
            backup = self.to_bytes()
            self.from_bytes(params[self.id])
        else:
            backup = None
        yield
        if backup is not None:
            self.from_bytes(backup)


class PyTorchWrapperRNN(PyTorchWrapper):
    """Wrap a PyTorch RNN model
    """

    def __call__(self, x_data, h_0=None):
        x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=False)
        # Make prediction
        out, h_n = self._model(x_var, h_0)
        return (self.ops.asarray(out.data), h_n)

    def begin_update(self, x_data, h_0=None, drop=0.0):
        """Return the output of the wrapped PyTorch model for the given input,
        along with a callback to handle the backward pass.
        """
        x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=True)

        # Make prediction
        out, h_n = self._model(x_var, h_0)
        # Shapes will be:
        # out = seq_len, batch, hidden_size * num_directions
        # h_n = num_layers * num_directions, batch, hidden_size

        def backward_pytorch_rnn(d_data, sgd=None):
            dy_data, _ = d_data
            dout = xp2torch(dy_data)
            torch.autograd.backward((out,), grad_tensors=(dout,))
            if sgd is not None:
                if self._optimizer is None:
                    self._optimizer = self._create_optimizer(sgd)
                self._optimizer.step()
                self._optimizer.zero_grad()
            return torch2xp(x_var.grad)

        return (torch2xp(out), h_n), backward_pytorch_rnn

    def resize_output(self, new_dim):
        # self.weight = nn.Parameter(F.pad(self.weight, ...)) # add classes
        # self.weight = nn.Parameter(F.pad(model.weight, ...)) # add classes
        raise NotImplementedError

    def resize_input(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        if self.id in params:
            backup = self.to_bytes()
            self.from_bytes(params[self.id])
        else:
            backup = None
        yield
        if backup is not None:
            self.from_bytes(backup)
