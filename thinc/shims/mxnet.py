from typing import Any
import contextlib
from io import BytesIO
import srsly
import tempfile
import copy

try:
    import mxnet.autograd
    import mxnet.optimizer
    import mxnet as mx
except ImportError:  # pragma: no cover
    pass

from ..util import mxnet2xp, xp2mxnet, convert_recursive, make_tempfile
from ..backends import get_current_ops
from ..types import ArgsKwargs
from .shim import Shim


class MXNetShim(Shim):
    """Interface between a MXNet model and a Thinc Model. This container is
    *not* a Thinc Model subclass itself.
    """

    def __call__(self, inputs, is_train):
        if is_train:
            return self.begin_update(inputs)
        else:
            return self.predict(inputs), lambda a: ...

    def predict(self, inputs: ArgsKwargs) -> Any:
        """Pass inputs through to the underlying MXNet model, and return the
        output. No conversions are performed. The MXNet model is set into
        evaluation mode.
        """
        mx.autograd.set_training(train_mode=False)
        with mxnet.autograd.pause():
            outputs = self._model(*inputs.args, **inputs.kwargs)
        mx.autograd.set_training(train_mode=True)
        return outputs

    def begin_update(self, inputs: ArgsKwargs):
        """Pass the inputs through to the underlying MXNet model, keeping
        track of which items in the input are tensors requiring gradients.
        If the model returns a single value, it is converted into a one-element
        tuple. Return the outputs and a callback to backpropagate.
        """
        mx.autograd.set_training(train_mode=True)
        mx.autograd.set_recording(True)
        output = self._model(*inputs.args, **inputs.kwargs)

        def backprop(grads):
            mx.autograd.set_recording(False)
            mxnet.autograd.backward(*grads.args, **grads.kwargs)
            return convert_recursive(
                lambda x: hasattr(x, "grad"), lambda x: x.grad, inputs
            )

        return output, backprop

    def finish_update(self, optimizer):
        if self._optimizer is None:
            self._optimizer, self._trainer = self._create_optimizer(optimizer)
        if getattr(optimizer, "max_grad_norm", None):
            mxnet.gluon.utils.clip_global_norm(
                self._model.parameters(), optimizer.max_grad_norm
            )
        self._trainer.step(1)
        for param in self._model.collect_params().values():
            param.zero_grad()
        self._update_mxnet_averages(optimizer)

    def _create_optimizer(self, sgd):
        if sgd.b1 != 0 and sgd.b2 != 0:
            optimizer = mxnet.optimizer.Adam(
                learning_rate=sgd.learn_rate, beta1=sgd.b1, beta2=sgd.b2
            )
        elif sgd.b2 == 0:
            optimizer = mxnet.optimizer.SGD(
                learning_rate=sgd.learn_rate, momentum=sgd.b1
            )
        else:
            raise NotImplementedError

        from mxnet import gluon

        return optimizer, gluon.Trainer(self._model.collect_params(), optimizer)

    def _update_mxnet_averages(self, sgd, *, init_steps=1):
        if getattr(sgd, "averages", None) is None:
            return
        # Collect parameters if we don't have them
        for name, param in self._model.collect_params().items():
            key = f"mxnet_{self.id}_{name}"
            sgd.nr_update[key] += 1
            xp_param = mxnet2xp(param.grad())
            if key in sgd.averages:
                sgd.ops.update_averages(sgd.averages[key], xp_param, sgd.nr_update[key])
            else:
                sgd.averages[key] = xp_param.copy()
                sgd.nr_update[key] = init_steps

    def copy(self, ctx: "mx.context.Context" = None):
        if ctx is None:
            ctx = mx.current_context()
        model_bytes = self.to_bytes()
        copied = copy.deepcopy(self)
        copied._model.initialize(ctx=ctx)
        copied.from_bytes(model_bytes)
        return copied

    def to_device(self, device):
        if device == "cpu":
            self._model = self.copy(mx.cpu())
        else:
            self._model = self.copy(mx.gpu())

    def to_bytes(self):
        # MXNet doesn't implement save/load without a filename
        with make_tempfile("w+b") as temp:
            self._model.save_parameters(temp.name)
            temp.seek(0)
            weights_bytes = temp.read()
        msg = {"config": self.cfg, "state": weights_bytes}
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data):
        msg = srsly.msgpack_loads(bytes_data)
        self.cfg = msg["config"]
        self._load_params(msg["state"])
        return self

    def _load_params(self, params):
        # MXNet doesn't implement save/load without a filename :(
        with make_tempfile("w+b") as temp:
            temp.write(params)
            self._model.load_parameters(temp.name, ctx=mx.current_context())
