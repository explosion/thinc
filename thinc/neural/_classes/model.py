# coding: utf8
from __future__ import unicode_literals, division

import numpy
import contextlib
from collections import OrderedDict
import srsly
import threading

from .. import util
from ..train import Trainer
from ..ops import NumpyOps, CupyOps
from ..mem import Memory
from ..util import get_ops, copy_array
from ... import check
from ...check import has_shape, is_sequence, is_float


class Model(object):
    """Model base class."""

    name = "model"
    id = 0
    lsuv = False
    ops = NumpyOps()
    Ops = NumpyOps
    Trainer = Trainer
    drop_factor = 1.0
    descriptions = []
    on_data_hooks = []
    on_init_hooks = []  # Use this to add layers
    _thread_local = threading.local()

    @classmethod
    @contextlib.contextmanager
    def define_operators(cls, operators):
        """Bind operators to specified functions for the scope of the context:

        Example
        -------

            model = Model()
            other = Model()
            with Model.define_operators({"+": lambda self, other: "plus"}):
                print(model + other)
                # "plus"
            print(model + other)
            # Raises TypeError --- binding limited to scope of with block.
        """
        curr_operators = dict(getattr(cls._thread_local, "operators", {}))
        cls._thread_local.operators = dict(operators)
        yield
        cls._thread_local.operators = dict(curr_operators)

    @classmethod
    @contextlib.contextmanager
    def use_device(cls, device):
        """Change the device to execute on for the scope of the block."""
        if device == cls.ops.device:
            yield
        else:
            curr_Ops = cls.Ops
            curr_ops = cls.ops
            cls.Ops = get_ops(device)
            cls.ops = cls.Ops()
            yield
            cls.Ops = curr_Ops
            cls.ops = curr_ops

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        self.name = self.__class__.name
        self.Ops = self.__class__.Ops
        self.ops = self.Ops()
        kwargs = self._update_defaults(args, kwargs)
        self._mem = Memory(self.ops)
        self._dims = {}
        if not hasattr(self, "_layers"):
            self._layers = []
        self.descriptions = dict(self.descriptions)
        self.on_init_hooks = list(self.on_init_hooks)
        self.on_data_hooks = list(self.on_data_hooks)

        for attr, install in self.descriptions.items():
            install(attr, self)
        for hook in self.on_init_hooks:
            hook(self, *args, **kwargs)
        self.set_id()

    def __getstate__(self):
        return srsly.pickle_dumps(self.__dict__)

    def __setstate__(self, state_data):
        self.__dict__ = srsly.pickle_loads(state_data)

    def _update_defaults(self, args, kwargs):
        new_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                new_kwargs[key] = value
        return new_kwargs

    def set_id(self):
        Model.id += 1
        self.id = Model.id
        for layer in self._layers:
            layer.set_id()

    # @check.args(equal_length)
    @check.arg(1, is_sequence)
    def begin_training(self, train_X, train_y=None, **trainer_cfg):
        for hook in self.on_data_hooks:
            hook(self, train_X, train_y)
        return self.Trainer(self, **trainer_cfg)

    @check.arg(2, is_float)
    @check.arg(1, has_shape(("nB", "nI")))
    def begin_update(self, X, drop=0.0):
        raise NotImplementedError

    def predict(self, X):
        y, _ = self.begin_update(X, drop=None)
        return y

    def predict_one(self, x):
        X = self.ops.expand_dims(x, axis=0)
        return self.predict(X)[0]

    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        backup = None
        weights = self._mem.weights
        if self.id in params:
            param = params[self.id]
            backup = weights.copy()
            copy_array(weights, param)
        if hasattr(self, "_layers"):
            contexts = [layer.use_params(params) for layer in self._layers]
            for context in contexts:
                next(context.gen)
        yield
        if backup is not None:
            copy_array(self._mem.weights, backup)
        for i, context in enumerate(contexts):
            # This is ridiculous, but apparently it's what you
            # have to do to make this work across Python 2/3?
            try:
                next(context.gen)
            except StopIteration:
                pass

    def __call__(self, x):
        """
        x
            Must match expected type
            Must match expected shape
        """
        return self.predict(x)

    def pipe(self, stream, batch_size=128):
        for batch in util.minibatch(stream, batch_size):
            ys = self.predict(batch)
            for y in ys:
                yield y

    def update(self, stream, batch_size=1000):
        for X, y in util.minibatch(stream, batch_size=batch_size):
            output, finish_update = self.begin_update(X)
            gradient = finish_update(y)
            yield gradient

    def walk(self):
        """Iterate out layers of the model, breadth-first."""
        queue = [self]
        seen = set()
        for node in queue:
            if node.id in seen:
                continue
            seen.add(node.id)
            yield node
            if hasattr(node, "_layers"):
                queue.extend(node._layers)

    def get_gradients(self):
        """Get non-zero gradients of the model's parameters, as a dictionary
        keyed by the parameter ID. The values are (weights, gradients) tuples.
        """
        gradients = {}
        for node in self.walk():
            if hasattr(node, "_mem") and node._mem.gradient.any():
                gradients[node.id] = [node._mem.weights, node._mem.gradient]
        return gradients

    def to_gpu(self, device_num):
        import cupy.cuda.device

        device = cupy.cuda.device.Device(device_num)
        device.use()
        queue = [self]
        for layer in queue:
            layer.ops = CupyOps()
            layer.Ops = CupyOps
            if hasattr(layer, "_mem"):
                layer._mem._mem = self.ops.xp.asarray(layer._mem._mem)
                layer._mem.ops = layer.ops
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return device

    def to_cpu(self):
        queue = [self]
        for layer in queue:
            layer.ops = NumpyOps()
            layer.Ops = NumpyOps
            if hasattr(layer, "_mem"):
                if hasattr(layer._mem._mem, "get"):
                    layer._mem._mem = layer._mem._mem.get()
                layer._mem.ops = layer.ops
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)

    def evaluate(self, X, y, batch_size=128):
        """
        x
            Must match expected type
            Must match expected shape
        y
            Must match expected type
        """
        scores = self.ops.flatten(list(self.pipe(X, batch_size=batch_size)))
        if not hasattr(y, "shape"):
            y = self.ops.flatten(y)
        scores = scores.reshape(y.shape)
        if len(scores.shape) == 1:
            correct = ((scores >= 0.5) == (y >= 0.5)).sum()
        else:
            correct = (scores.argmax(axis=1) == y.argmax(axis=1)).sum()
        return correct / y.shape[0]

    def evaluate_logloss(self, X, y, minimum=None, maximum=None):
        yh = self.ops.xp.vstack(self.pipe(X))
        yh = yh.reshape(y.shape)
        if minimum is not None:
            yh = self.ops.xp.maximum(yh, minimum)
        if maximum is not None:
            yh = self.ops.xp.minimum(yh, maximum)
        assert len(yh.shape) == 1
        losses = -y * self.ops.xp.log(yh + 1e-8) - (1 - y) * self.ops.xp.log(
            (1 - yh) + 1e-8
        )
        return losses.mean()

    @check.operator_is_defined("+")
    def __add__(self, other):
        """Apply the function bound to the '+' operator."""
        return self._thread_local.operators["+"](self, other)

    @check.operator_is_defined("-")
    def __sub__(self, other):
        """Apply the function bound to the '-' operator."""
        return self._thread_local.operators["-"](self, other)

    @check.operator_is_defined("*")
    def __mul__(self, other):
        """Apply the function bound to the '*' operator."""
        return self._thread_local.operators["*"](self, other)

    @check.operator_is_defined("@")
    def __matmul__(self, other):
        """Apply the function bound to the '@' operator."""
        return self._thread_local.operators["@"](self, other)

    @check.operator_is_defined("/")
    def __div__(self, other):
        """Apply the function bound to the '/' operator."""
        return self._thread_local.operators["/"](self, other)

    @check.operator_is_defined("/")
    def __truediv__(self, other):  # pragma: no cover
        """Apply the function bound to the '/' operator."""
        return self._thread_local.operators["/"](self, other)

    @check.operator_is_defined("//")
    def __floordiv__(self, other):
        """Apply the function bound to the '//' operator."""
        return self._thread_local.operators["//"](self, other)

    @check.operator_is_defined("%")
    def __mod__(self, other):
        """Apply the function bound to the '%' operator."""
        return self._thread_local.operators["%"](self, other)

    @check.operator_is_defined("**")
    def __pow__(self, other, modulo=None):
        """Apply the function bound to the '**' operator."""
        return self._thread_local.operators["**"](self, other)

    @check.operator_is_defined("<<")
    def __lshift__(self, other):
        """Apply the function bound to the '<<' operator."""
        return self._thread_local.operators["<<"](self, other)

    @check.operator_is_defined(">>")
    def __rshift__(self, other):
        """Apply the function bound to the '>>' operator."""
        return self._thread_local.operators[">>"](self, other)

    @check.operator_is_defined("&")
    def __and__(self, other):
        """Apply the function bound to the '&' operator."""
        return self._thread_local.operators["&"](self, other)

    @check.operator_is_defined("^")
    def __xor__(self, other):
        """Apply the function bound to the '^' operator."""
        return self._thread_local.operators["^"](self, other)

    @check.operator_is_defined("|")
    def __or__(self, other):
        """Apply the function bound to the '|' operator."""
        return self._thread_local.operators["|"](self, other)

    def to_bytes(self):
        weights = []
        queue = [self]
        i = 0
        for layer in queue:
            # Hack to support saving/loading PyTorch models. TODO: Improve
            if hasattr(layer, "_model") and not isinstance(layer._model, Model):
                weights.append(layer.to_bytes())
            elif hasattr(layer, "_mem"):
                weights.append(
                    OrderedDict(
                        (
                            (b"dims", OrderedDict(sorted(layer._dims.items()))),
                            (b"params", []),
                        )
                    )
                )
                if hasattr(layer, "seed"):
                    weights[-1][b"seed"] = layer.seed

                offsets = sorted(layer._mem._offsets.items())
                for (id_, name), (start, row, shape) in offsets:
                    if row == 1:
                        continue
                    param = layer._mem.get((id_, name))
                    if not isinstance(layer._mem.weights, numpy.ndarray):
                        param = param.get()
                    weights[-1][b"params"].append(
                        OrderedDict(
                            (
                                (b"name", name),
                                (b"offset", start),
                                (b"shape", shape),
                                (b"value", param),
                            )
                        )
                    )
                i += 1
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return srsly.msgpack_dumps({b"weights": weights})

    def from_bytes(self, bytes_data):
        data = srsly.msgpack_loads(bytes_data)
        weights = data[b"weights"]
        queue = [self]
        i = 0
        for layer in queue:
            # Hack to support saving/loading PyTorch models. TODO: Improve
            if hasattr(layer, "_model") and not isinstance(layer._model, Model):
                layer.from_bytes(weights[i])
                i += 1
            elif hasattr(layer, "_mem"):
                if b"seed" in weights[i]:
                    layer.seed = weights[i][b"seed"]
                for dim, value in weights[i][b"dims"].items():
                    if isinstance(dim, bytes):
                        dim = dim.decode("utf8")
                    setattr(layer, dim, value)
                for param in weights[i][b"params"]:
                    name = param[b"name"]
                    if isinstance(name, bytes):
                        name = name.decode("utf8")
                    dest = getattr(layer, name)
                    copy_array(dest, param[b"value"])
                i += 1
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return self

    def to_disk(self, path):
        path = util.ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes())

    def from_disk(self, path):
        path = util.ensure_path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        return self.from_bytes(bytes_data)
