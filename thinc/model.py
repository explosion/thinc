from typing import Dict, List, Callable, Optional, Any, Union, Iterable, Set
import numpy
import contextlib
import srsly
from pathlib import Path
import copy

from .backends import NumpyOps, CupyOps, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .mem import Memory
from .util import copy_array, get_width, create_thread_local
from .types import Array


def create_init(initializers: Dict[str, Callable]) -> Callable:
    """Create an init function, given a dictionary of parameter initializers."""

    def init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if X is not None:
            model.set_dim("nI", get_width(X))
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        W = model.ops.allocate((model.get_dim("nO"), model.get_dim("nI")))
        b = model.ops.allocate((model.get_dim("nO"),))
        if "W" in initializers:
            initializers["W"](W, inplace=True)
        if "b" in initializers:
            initializers["b"](b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)

    return init


class Model:
    """Base class for Thinc models and layers."""

    global_id: int = 0
    _thread_local = create_thread_local({"operators": {}})

    name: str
    ops: Union[NumpyOps, CupyOps]
    id: int
    _func: Callable
    _init: Callable
    _mem: Memory
    _params: Dict[str, Optional[bool]]
    _dims: Dict[str, Optional[int]]
    _grads: Dict[str, Optional[bool]]
    _layers: List["Model"]
    _attrs: Dict[str, Any]

    # This "locks" the class, so we get an error if you try to assign to
    # an unexpected variable.
    __slots__ = [
        "name",
        "id",
        "ops",
        "_func",
        "_init",
        "_mem",
        "_params",
        "_dims",
        "_grads",
        "_layers",
        "_attrs",
    ]

    def __init__(
        self,
        name: str,
        forward: Callable,
        *,
        init: Callable = lambda *a, **k: None,
        dims: Dict[str, Optional[int]] = {},
        params: Dict[str, Optional[Array]] = {},
        grads: Dict[str, Optional[Array]] = {},
        layers: List["Model"] = [],
        attrs: Dict[str, object] = {},
        ops: Optional[Union[NumpyOps, CupyOps]] = None,
    ):
        self.name = name
        # Assign to callable attrs: https://github.com/python/mypy/issues/2427
        setattr(self, "_func", forward)
        setattr(self, "_init", init)
        self.ops = ops if ops is not None else get_current_ops()
        self._mem = Memory(self.ops)
        self._dims = dict(dims)
        self._attrs = dict(attrs)
        self._layers = list(layers)
        self.__class__.global_id += 1
        self.id = self.__class__.global_id
        self._params = {}
        self._grads = {}
        for name, value in params.items():
            self._params[name] = None
            if value is not None:
                self.set_param(name, value)
        for name, value in grads.items():
            self._grads[name] = None
            if value is not None:
                self.set_grad(name, value)

    @property
    def layers(self):
        return self._layers

    @classmethod
    @contextlib.contextmanager
    def define_operators(cls, operators):
        """Bind operators to specified functions for the scope of the context:

        Example:
            model = Model()
            other = Model()
            with Model.define_operators({"+": lambda self, other: "plus"}):
                print(model + other)
                # "plus"
            print(model + other)
            # Raises TypeError --- binding limited to scope of with block.
        """
        curr_operators = dict(cls._thread_local.operators)
        cls._thread_local.operators = dict(operators)
        yield
        cls._thread_local.operators = dict(curr_operators)

    def dim_is_unset(self, name: str) -> bool:
        return self.has_dim(name) and self.get_dim(name) is None

    def has_dim(self, name: str) -> bool:
        """Check whether the model has a dimension of a given name."""
        return name in self._dims

    def get_dim(self, name: str) -> int:
        """Retrieve the value of a dimension of the given name, or None if unset."""
        if name not in self._dims:
            raise KeyError(f"Can't get dimension '{name}'")
        return self._dims[name]

    def set_dim(self, name: str, value: int) -> None:
        """Set a value for a dimension."""
        if name not in self._dims:
            raise KeyError(f"Can't set dimension '{name}'")
        self._dims[name] = value

    def has_param(self, name: str) -> bool:
        """Check whether the model has a weights parameter of the given name."""
        return name in self._params

    def get_param(self, name: str) -> Array:
        """Retrieve a weights parameter by name."""
        if name not in self._params:
            raise KeyError(f"Unknown param: {name}")
        key = (self.id, name)
        if key not in self._mem:
            raise KeyError(f"Parameter '{name}' as not been allocated yet")
        return self._mem[key]

    def set_param(self, name: str, value: Array) -> None:
        """Set a weights parameter's value."""
        key = (self.id, name)
        if key not in self._mem:
            self._mem.add(key, value.shape)
        data = self._mem.get((self.id, name))
        copy_array(dst=data, src=value)
        self._params[name] = True

    def inc_grad(self, param_name: str, value: Array) -> None:
        """Check whether the model has a gradient of the given name."""
        grad_name = f"d_{param_name}"
        key = (self.id, grad_name)
        param_key = (self.id, param_name)
        if key in self._mem:
            grad = self._mem.get(key)
        else:
            grad = self._mem.add_gradient(key, param_key)
        grad += value
        self._grads[grad_name] = True

    def get_grad(self, param_name: str) -> Array:
        """Get a gradient from the model."""
        grad_name = f"d_{param_name}"
        key = (self.id, grad_name)
        if key not in self._mem:
            raise KeyError(f"Gradient '{grad_name}' as not been allocated yet")
        return self._mem[key]

    def set_grad(self, param_name: str, value: Array) -> None:
        """Set a gradient value for the model."""
        grad_name = f"d_{param_name}"
        data = self._mem.get((self.id, grad_name))
        copy_array(dst=data, src=value)

    def has_attr(self, name: str) -> bool:
        """Check whether the model has the given attribute."""
        return name in self._attrs

    def get_attr(self, name: str) -> Any:
        """Get the attribute, or None if not present."""
        if name not in self._attrs:
            raise KeyError(f"Can't get attribute '{name}'")
        return self._attrs[name]

    def set_attr(self, name: str, value: Any) -> None:
        if name not in self._attrs:
            raise KeyError(f"Can't set attribute '{name}'")
        self._attrs[name] = value

    def __call__(self, X, is_train=False):
        return self._func(self, X, is_train=is_train)

    def initialize(self, X=None, Y=None):
        if self._init is not None:
            self._init(self, X=X, Y=Y)

    def begin_update(self, X):
        """Run the model over a batch of data, returning the output and a callback
        to complete the backward pass.

        X: A batch of input data.

        RETURNS:
            A tuple (Y, finish_update), where Y is a batch of output data,
            and finish_update is a callback that takes the gradient with
            respect to the output and an optimizer function, and returns
            the gradient with respect to the input.
        """
        return self._func(self, X, is_train=True)

    def predict(self, X):
        return self._func(self, X, is_train=False)[0]

    def finish_update(self, optimizer: Optimizer) -> None:
        """Update parameters with current gradients.

        optimizer (Callable[array, array, key=None]):
            The optimizer. The function is called with each parameter and
            gradient of the model.
        """
        optimizer(self._mem.weights, self._mem.gradient, key=self.id)
        seen = set([self.id])
        for node in self.walk():
            if node.id not in seen:
                node.finish_update(optimizer)
                seen.add(node.id)

    def set_child_attrs(self, name: str, attr: str, value) -> int:
        """Walk through layers for any that match the given name, and set
        an attribute on those nodes.

        >>> node.walk_set_attrs("dropout", "rate", 0.2)

        name (str): The node name to look for.
        attr (str): The attribute to set.
        value (object): The value to set.

        RETURNS (int): The number of matched nodes.
        """
        n_set = 0
        for node in self.walk():
            if node.name == name:
                node.set_attr(attr, value)
                n_set += 1
        return n_set

    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        """Context manager to temporarily set the model's parameters to specified
        values.

        params (dict): A dictionary keyed by model IDs, whose values are arrays
            of weight values.
        """
        backup = None
        weights = self._mem.weights
        if self.id in params:
            param = params[self.id]
            backup = weights.copy()
            copy_array(dst=weights, src=param)
        if hasattr(self, "_layers"):
            contexts = [layer.use_params(params) for layer in self._layers]
            for context in contexts:
                next(context.gen)
        yield
        if backup is not None:
            copy_array(dst=self._mem.weights, src=backup)
        for i, context in enumerate(contexts):
            # This is ridiculous, but apparently it's what you
            # have to do to make this work across Python 2/3?
            try:
                next(context.gen)
            except StopIteration:
                pass

    def walk(self) -> Iterable["Model"]:
        """Iterate out layers of the model, breadth-first."""
        queue = [self]
        seen: Set[int] = set()
        for node in queue:
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            if hasattr(node, "_layers"):
                queue.extend(node._layers)

    def get_gradients(self) -> Dict[int, Array]:
        """Get non-zero gradients of the model's parameters, as a dictionary
        keyed by the parameter ID. The values are (weights, gradients) tuples.
        """
        gradients = {}
        for node in self.walk():
            if hasattr(node, "_mem") and node._mem.gradient.any():
                gradients[node.id] = [node._mem.weights, node._mem.gradient]
        return gradients

    def copy(self) -> "Model":
        copied = Model(
            self.name,
            self._func,
            init=self._init,
            params=copy.deepcopy(self._params),
            grads=copy.deepcopy(self._grads),
            dims=copy.deepcopy(self._dims),
            attrs=copy.deepcopy(self._attrs),
            layers=[layer.copy() for layer in self._layers]
        )
        for name, is_allocated in self._params.items():
            if is_allocated:
                copied.set_param(name, self.get_param(name))
        for name, is_allocated in self._grads.items():
            if is_allocated:
                copied.set_grad(name, self.get_grad(name))
        return copied

    def to_gpu(self, device_num: int) -> None:
        """Transfer the model to a given GPU device."""
        import cupy.cuda.device

        device = cupy.cuda.device.Device(device_num)
        device.use()
        queue = [self]
        for layer in queue:
            layer.ops = CupyOps()
            if hasattr(layer, "_mem"):
                layer._mem._mem = self.ops.xp.asarray(layer._mem._mem)
                layer._mem.ops = layer.ops
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return device

    def to_cpu(self) -> None:
        """Copy the model to CPU."""
        queue = [self]
        for layer in queue:
            layer.ops = NumpyOps()
            if hasattr(layer, "_mem"):
                if hasattr(layer._mem._mem, "get"):
                    layer._mem._mem = layer._mem._mem.get()
                layer._mem.ops = layer.ops
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)

    def to_bytes(self) -> bytes:
        """Serialize the model to a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        weights = []
        queue = [self]
        i = 0
        for layer in queue:
            # Hack to support saving/loading PyTorch models. TODO: Improve
            if hasattr(layer, "_model") and not isinstance(layer._model, self):
                weights.append(layer.to_bytes())
            elif hasattr(layer, "_mem"):
                weights.append(
                    {
                        b"dims": dict(sorted(layer._dims.items())),
                        b"params": [],
                        b"attrs": dict(sorted(layer._attrs.items())),
                    }
                )
                offsets = sorted(layer._mem._offsets.items())
                for (id_, name), (start, row, shape) in offsets:
                    if row == 1:
                        continue
                    param = layer._mem.get((id_, name))
                    if not isinstance(layer._mem.weights, numpy.ndarray):
                        param = param.get()
                    weights[-1][b"params"].append(
                        {
                            b"name": name,
                            b"offset": start,
                            b"shape": shape,
                            b"value": param,
                        }
                    )
                i += 1
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return srsly.msgpack_dumps({b"weights": weights})

    def from_bytes(self, bytes_data: bytes) -> "Model":
        """Deserialize the model from a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        data = srsly.msgpack_loads(bytes_data)
        weights = data[b"weights"]
        queue = [self]
        i = 0
        for layer in queue:
            # Hack to support saving/loading PyTorch models. TODO: Improve
            if hasattr(layer, "_model") and not isinstance(layer._model, "Model"):
                layer.from_bytes(weights[i])
                i += 1
            elif hasattr(layer, "_mem"):
                for attr, value in weights[i][b"attrs"].items():
                    layer.set_attr(attr, value)
                for dim, value in weights[i][b"dims"].items():
                    if isinstance(dim, bytes):
                        dim = dim.decode("utf8")
                    layer.set_dim(dim, value)
                for param in weights[i][b"params"]:
                    name = param[b"name"]
                    if isinstance(name, bytes):
                        name = name.decode("utf8")
                    layer.set_param(name, param[b"value"])
                i += 1
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return self

    def to_disk(self, path: Union[Path, str]) -> None:
        """Serialize the model to disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        path = Path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes())

    def from_disk(self, path: Union[Path, str]) -> "Model":
        """Deserialize the model from disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        path = Path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        return self.from_bytes(bytes_data)

    def __add__(self, other) -> "Model":
        """Apply the function bound to the '+' operator."""
        if "+" not in self._thread_local.operators:
            raise TypeError("Undefined operator: +")
        return self._thread_local.operators["+"](self, other)

    def __sub__(self, other) -> "Model":
        """Apply the function bound to the '-' operator."""
        if "-" not in self._thread_local.operators:
            raise TypeError("Undefined operator: -")
        return self._thread_local.operators["-"](self, other)

    def __mul__(self, other) -> "Model":
        """Apply the function bound to the '*' operator."""
        if "*" not in self._thread_local.operators:
            raise TypeError("Undefined operator: *")
        return self._thread_local.operators["*"](self, other)

    def __matmul__(self, other) -> "Model":
        """Apply the function bound to the '@' operator."""
        if "@" not in self._thread_local.operators:
            raise TypeError("Undefined operator: @")
        return self._thread_local.operators["@"](self, other)

    def __div__(self, other) -> "Model":
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __truediv__(self, other) -> "Model":  # pragma: no cover
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __floordiv__(self, other) -> "Model":
        """Apply the function bound to the '//' operator."""
        if "//" not in self._thread_local.operators:
            raise TypeError("Undefined operator: //")
        return self._thread_local.operators["//"](self, other)

    def __mod__(self, other) -> "Model":
        """Apply the function bound to the '%' operator."""
        if "%" not in self._thread_local.operators:
            raise TypeError("Undefined operator: %")
        return self._thread_local.operators["%"](self, other)

    def __pow__(self, other, modulo=None) -> "Model":
        """Apply the function bound to the '**' operator."""
        if "**" not in self._thread_local.operators:
            raise TypeError("Undefined operator: **")
        return self._thread_local.operators["**"](self, other)

    def __lshift__(self, other: "Model") -> "Model":
        """Apply the function bound to the '<<' operator."""
        if "<<" not in self._thread_local.operators:
            raise TypeError("Undefined operator: <<")
        return self._thread_local.operators["<<"](self, other)

    def __rshift__(self, other) -> "Model":
        """Apply the function bound to the '>>' operator."""
        if ">>" not in self._thread_local.operators:
            raise TypeError("Undefined operator: >>")
        return self._thread_local.operators[">>"](self, other)

    def __and__(self, other) -> "Model":
        """Apply the function bound to the '&' operator."""
        if "&" not in self._thread_local.operators:
            raise TypeError("Undefined operator: &")
        return self._thread_local.operators["&"](self, other)

    def __xor__(self, other) -> "Model":
        """Apply the function bound to the '^' operator."""
        if "^" not in self._thread_local.operators:
            raise TypeError("Undefined operator: ^")
        return self._thread_local.operators["^"](self, other)

    def __or__(self, other) -> "Model":
        """Apply the function bound to the '|' operator."""
        if "|" not in self._thread_local.operators:
            raise TypeError("Undefined operator: |")
        return self._thread_local.operators["|"](self, other)
