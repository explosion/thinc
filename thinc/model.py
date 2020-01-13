from typing import Dict, List, Callable, Optional, Any, Union, Iterable, Set
from typing import Generic, Sequence, Tuple, TypeVar
import numpy
import contextlib
import srsly
from pathlib import Path
import copy

from .backends import NumpyOps, CupyOps, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .backends.mem import Memory
from .shims import Shim
from .util import copy_array, get_width, create_thread_local
from .types import Array


InT = TypeVar("InT")
OutT = TypeVar("OutT")


def create_init(initializers: Dict[str, Callable]) -> Callable:
    """Create an init function, given a dictionary of parameter initializers."""

    def init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if X is not None:
            model.set_dim("nI", get_width(X))
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        W = model.ops.alloc_f2d(model.get_dim("nO"), model.get_dim("nI"))
        b = model.ops.alloc_f1d(model.get_dim("nO"))
        if "W" in initializers:
            initializers["W"](W, inplace=True)
        if "b" in initializers:
            initializers["b"](b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)

    return init


class Model(Generic[InT, OutT]):
    """Class for implementing Thinc models and layers."""

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
    _shims: List[Shim]
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
        "_attrs",
        "_refs",
        "_layers",
        "_shims",
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
        layers: Sequence["Model"] = [],
        shims: List[Shim] = [],
        attrs: Dict[str, Any] = {},
        refs: Dict[str, Optional["Model"]] = {},
        ops: Optional[Union[NumpyOps, CupyOps]] = None,
    ):
        """Initialize a new model."""
        self.name = name
        # Assign to callable attrs: https://github.com/python/mypy/issues/2427
        setattr(self, "_func", forward)
        setattr(self, "_init", init)
        self.ops = ops if ops is not None else get_current_ops()
        self._mem = Memory(self.ops)
        self._dims = dict(dims)
        self._attrs = dict(attrs)
        self._refs = dict(refs)
        self._layers = list(layers)
        self._shims = list(shims)
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
    def layers(self) -> List["Model"]:
        """A list of child layers of the model. You can append to it to add
        layers but not reassign it.
        """
        return self._layers

    @property
    def shims(self) -> List[Shim]:
        return self._shims

    @property
    def param_names(self) -> Tuple[str, ...]:
        """Get the names of registered parameter (including unset)."""
        return tuple(self._params.keys())

    @property
    def grad_names(self) -> Tuple[str, ...]:
        """Get the names of parameters with registered gradients (including unset)."""
        return tuple([name for name in self.param_names if self.has_grad(name)])

    @property
    def dim_names(self) -> Tuple[str, ...]:
        """Get the names of registered dimensions (including unset)."""
        return tuple(self._dims.keys())

    @property
    def attr_names(self) -> Tuple[str, ...]:
        """Get the names of attributes."""
        return tuple(self._attrs.keys())

    @property
    def ref_names(self) -> Tuple[str, ...]:
        """Get the names of registered node references (including unset)."""
        return tuple(self._refs.keys())

    @classmethod
    @contextlib.contextmanager
    def define_operators(cls, operators: Dict[str, Callable]):
        """Bind arbitrary binary functions to Python operators, for use in any
        `Model` instance. Can (and should) be used as a contextmanager.

        EXAMPLE:
            with Model.define_operators({">>": chain}):
                model = ReLu(512) >> ReLu(512) >> Softmax()
        """
        curr_operators = dict(cls._thread_local.operators)
        cls._thread_local.operators = dict(operators)
        yield
        cls._thread_local.operators = dict(curr_operators)

    def has_dim(self, name: str) -> Optional[bool]:
        """Check whether the model has a dimension of a given name. If the
        dimension is registered but the value is unset, returns None.
        """
        if name not in self._dims:
            return False
        elif self._dims[name] is not None:
            return True
        else:
            return None

    def get_dim(self, name: str) -> int:
        """Retrieve the value of a dimension of the given name."""
        if name not in self._dims:
            raise KeyError(f"Cannot get dimension '{name}' for model '{self.name}'")
        value = self._dims[name]
        if value is None:
            err = f"Cannot get dimension '{name}' for model '{self.name}': value unset"
            raise ValueError(err)
        else:
            return value

    def set_dim(self, name: str, value: int) -> None:
        """Set a value for a dimension."""
        if name not in self._dims:
            raise KeyError(f"Cannot set dimension '{name}' for model '{self.name}'.")
        old_value = self._dims[name]
        if old_value is not None and old_value != value:
            err = f"Attempt to change dimension '{name}' for model '{self.name}' from {old_value} to {value}"
            raise ValueError(err)
        self._dims[name] = value

    def has_param(self, name: str) -> Optional[bool]:
        """Check whether the model has a weights parameter of the given name.

        Returns None if the parameter is registered but currently unset.
        """
        if name not in self._params:
            return False
        elif self._params[name] is not None:
            return True
        else:
            return None

    def get_param(self, name: str) -> Array:
        """Retrieve a weights parameter by name."""
        if name not in self._params:
            raise KeyError(f"Unknown param: '{name}' for model '{self.name}'.")
        key = (self.id, name)
        if key not in self._mem:
            raise KeyError(
                f"Parameter '{name}' for model '{self.name}' has not been allocated yet."
            )
        return self._mem[key]

    def set_param(self, name: str, value: Optional[Array]) -> None:
        """Set a weights parameter's value."""
        if value is None:
            self._params[name] = None
        else:
            key = (self.id, name)
            if key not in self._mem:
                self._mem.add(key, value.shape)
            data = self._mem[(self.id, name)]
            try:
                copy_array(dst=data, src=value)
            except ValueError as e:  # pragma: no cover
                err = f"Cannot set param '{name}' for model '{self.name}': {e}"
                raise ValueError(err)
            self._params[name] = True

    def inc_grad(self, name: str, value: Array) -> None:
        """Check whether the model has a gradient of the given name."""
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        param_key = (self.id, name)
        if key in self._mem:
            grad = self._mem[key]
        else:
            grad = self._mem.add_gradient(key, param_key)
        if grad.shape != value.shape:
            raise ValueError(
                f"Shape mismatch: Cannot add a value to the gradient of param "
                f"'{name}' for model '{self.name}'. Got: {grad.shape} for "
                f"original gradient and {value.shape} for value to be added"
            )
        grad += value
        self._grads[grad_name] = True

    def has_grad(self, name: str) -> Optional[bool]:
        """Check whether the model has a non-zero gradient for a parameter.
        Returns None if the gradient is allocated but currently 0.
        """
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        if key not in self._mem:
            return False
        elif not self._mem[key].any():
            return None
        else:
            return True

    def get_grad(self, name: str) -> Array:
        """Get a gradient from the model."""
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        if key not in self._mem:
            err = f"Gradient '{grad_name}' has not been allocated yet for model '{self.name}'"
            raise KeyError(err)
        return self._mem[key]

    def set_grad(self, name: str, value: Array) -> None:
        """Set a gradient value for the model."""
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        if key not in self._mem:
            self.inc_grad(name, value)
        else:
            data = self._mem[key]
            try:
                copy_array(dst=data, src=value)
            except ValueError as e:  # pragma: no cover
                err = f"Cannot set grad '{grad_name}' for model '{self.name}': {e}"
                raise ValueError(err)

    def has_attr(self, name: str) -> bool:
        """Check whether the model has the given attribute."""
        return name in self._attrs

    def get_attr(self, name: str) -> Any:
        """Get the attribute. Raises KeyError if not present."""
        if name not in self._attrs:
            raise KeyError(f"Cannot get attribute '{name}' for model '{self.name}'.")
        return self._attrs[name]

    def set_attr(self, name: str, value: Any) -> None:
        """Set the attribute to the given value."""
        self._attrs[name] = value

    def has_ref(self, name: str) -> Optional[bool]:
        """Check whether the model has a reference of a given name. If the
        reference is registered but the value is unset, returns None.
        """
        if name not in self._refs:
            return False
        elif self._refs[name] is not None:
            return True
        else:
            return None

    def get_ref(self, name: str) -> "Model":
        """Retrieve the value of a reference of the given name."""
        if name not in self._refs:
            raise KeyError(f"Cannot get reference '{name} for model '{self.name}'.")
        value = self._refs[name]
        if value is None:
            err = f"Cannot get reference '{name}' for model '{self.name}': value unset."
            raise ValueError(err)
        else:
            return value

    def set_ref(self, name: str, value: Optional["Model"]) -> None:
        """Set a value for a reference."""
        if value is None:
            self._refs[name] = value
        elif value in self.walk():
            self._refs[name] = value
        else:
            raise ValueError("Cannot add reference to node not in tree.")

    def __call__(self, X: InT, is_train: bool) -> Tuple[OutT, Callable]:
        """Call the model's `forward` function, returning the output and a
        callback to compute the gradients via backpropagation."""
        return self._func(self, X, is_train=is_train)

    def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":
        """Finish initialization of the model, optionally providing a batch of
        example input and output data to perform shape inference."""
        if self._init is not None:
            self._init(self, X=X, Y=Y)
        return self

    def begin_update(self, X: InT) -> Tuple[OutT, Callable[[InT], OutT]]:
        """Run the model over a batch of data, returning the output and a
        callback to complete the backward pass. A tuple (Y, finish_update),
        where Y is a batch of output data, and finish_update is a callback that
        takes the gradient with respect to the output and an optimizer function,
        and returns the gradient with respect to the input.
        """
        return self._func(self, X, is_train=True)

    def predict(self, X: InT) -> OutT:
        """Call the model's `forward` function with `is_train=False`, and return
        only the output, instead of the `(output, callback)` tuple.
        """
        return self._func(self, X, is_train=False)[0]

    def finish_update(self, optimizer: Optimizer) -> None:
        """Update parameters with current gradients. The optimizer is called
        with each parameter and gradient of the model.
        """
        seen = set()
        for node in self.walk():
            if node.id not in seen:
                # Kind of ugly to use the _mem.weights -- would make more sense
                # to call node.finish_update. Maybe we could pass in a set
                # of visited?
                optimizer(node._mem.weights, node._mem.gradient, key=node.id)
                seen.add(node.id)
                for shim in node.shims:
                    shim.finish_update(optimizer)

    @contextlib.contextmanager
    def use_params(self, params: Dict[int, Array]):
        """Context manager to temporarily set the model's parameters to
        specified values. The params are a dictionary keyed by model IDs, whose
        values are arrays of weight values.
        """
        backup = None
        weights = self._mem.weights
        if self.id in params:
            param = params[self.id]
            backup = weights.copy()
            copy_array(dst=weights, src=param)
        with contextlib.ExitStack() as stack:
            for layer in self.layers:
                stack.enter_context(layer.use_params(params))
            for shim in self.shims:
                stack.enter_context(shim.use_params(params))
            yield
        if backup is not None:
            copy_array(dst=self._mem.weights, src=backup)

    def walk(self) -> Iterable["Model"]:
        """Iterate out layers of the model, breadth-first."""
        queue = [self]
        seen: Set[int] = set()
        for node in queue:
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            queue.extend(node.layers)

    def remove_node(self, node: "Model") -> None:
        """Remove a node from all layers lists, and then update references.
        References that no longer point to a node within the tree will be set
        to `None`. For instance, let's say a node has its grandchild as a reference.
        If the child is removed, the grandchild reference will be left dangling,
        so will be set to None.
        """
        for child in list(self.walk()):
            while node in child.layers:
                child.layers.remove(node)
        tree = set(self.walk())
        for node in tree:
            for name in node.ref_names:
                ref = node.get_ref(name)
                if ref is not None and ref not in tree:
                    node.set_ref(name, None)

    def get_gradients(self) -> Dict[int, Tuple[Array, Array]]:
        """Get non-zero gradients of the model's parameters, as a dictionary
        keyed by the parameter ID. The values are (weights, gradients) tuples.
        """
        gradients = {}
        for node in self.walk():
            if hasattr(node, "_mem") and node._mem.gradient.any():
                gradients[node.id] = (node._mem.weights, node._mem.gradient)
        return gradients

    def copy(self) -> "Model":
        """
        Create a copy of the model, its attributes, and its parameters. Any child
        layers will also be deep-copied. The copy will receive a distinct `model.id`
        value.
        """
        params = {}
        for key, value in self._params.items():
            params[key] = None if value is None else self.get_param(key)
        grads = {}
        for key, value in self._grads.items():
            grads[key] = None if value is None else self.get_grad(key)
        copied: Model[InT, OutT] = Model(
            self.name,
            self._func,
            init=self._init,
            params=copy.deepcopy(params),
            grads=copy.deepcopy(grads),
            dims=copy.deepcopy(self._dims),
            attrs=copy.deepcopy(self._attrs),
            layers=[layer.copy() for layer in self.layers],
        )
        # The `_params` and `_grads` dicts don't hold the actual values --
        # those are within the `model._mem` object. So we need to call `set_param`
        # on the copy.
        for name, is_allocated in self._params.items():
            if is_allocated:
                copied.set_param(name, self.get_param(name))
        for name, is_allocated in self._grads.items():
            if is_allocated:
                copied.set_grad(name, self.get_grad(name))
        return copied

    def to_gpu(self, gpu_id: int) -> None:  # pragma: no cover
        """Transfer the model to a given GPU device."""
        import cupy.cuda.device

        device = cupy.cuda.device.Device(gpu_id)
        device.use()
        for layer in self.walk():
            layer.ops = CupyOps()
            if hasattr(layer, "_mem"):
                layer._mem._mem = self.ops.xp.asarray(layer._mem._mem)
                layer._mem.ops = layer.ops
        return device

    def to_cpu(self) -> None:
        """Copy the model to CPU."""
        for layer in self.walk():
            layer.ops = NumpyOps()
            if hasattr(layer, "_mem"):
                if hasattr(layer._mem._mem, "get"):
                    layer._mem._mem = layer._mem._mem.get()
                layer._mem.ops = layer.ops

    def to_bytes(self) -> bytes:
        """Serialize the model to a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        weights: List[Union[str, Dict[str, Any]]] = []
        nodes = list(self.walk())
        # Serialize references by their index into the flattened tree.
        # This is the main reason we can't accept out-of-tree references:
        # we'd have no way to serialize/deserialize them.
        node_to_i: Dict[Optional[Model], Optional[int]]
        node_to_i = {node: i for i, node in enumerate(nodes)}
        # We also need an entry 'None', as references can be set to None.
        node_to_i[None] = None
        for i, layer in enumerate(nodes):
            # Separate attrs that need to be serialized/deserialized with
            # to_/from_bytes.
            obj_attrs = {}
            flat_attrs = {}
            for name, value in layer._attrs.items():
                if type(value) not in (str, int, float, bool) and hasattr(
                    value, "to_bytes"
                ):
                    obj_attrs[name] = value.to_bytes()
                else:
                    flat_attrs[name] = value

            refs = {name: node_to_i[ref] for name, ref in layer._refs.items()}
            weights.append(
                {
                    "dims": layer._dims,
                    "params": [],
                    "obj_attrs": obj_attrs,
                    "flat_attrs": flat_attrs,
                    "shims": [shim.to_bytes() for shim in layer.shims],
                    "refs": refs,
                }
            )
            for (id_, name), (start, row, shape) in layer._mem._offsets.items():
                if row == 1:
                    continue
                param = layer._mem[(id_, name)]
                if not isinstance(
                    layer._mem.weights, numpy.ndarray
                ):  # pragma: no cover
                    param = param.get()
                weights[-1]["params"].append(  # type: ignore
                    {"name": name, "offset": start, "shape": shape, "value": param}
                )
        return srsly.msgpack_dumps({"weights": weights})

    def from_bytes(self, bytes_data: bytes) -> "Model":
        """Deserialize the model from a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        msg = srsly.msgpack_loads(bytes_data)
        nodes = list(self.walk())
        if len(msg["weights"]) != len(nodes):
            raise ValueError("Cannot deserialize model: mismatched structure.")
        for layer, data in zip(nodes, msg["weights"]):
            for attr, value in data["flat_attrs"].items():
                layer.set_attr(attr, value)
            for attr, value in data["obj_attrs"].items():
                layer.get_attr(attr).from_bytes(value)
            for dim, value in data["dims"].items():
                layer.set_dim(dim, value)
            for param in data["params"]:
                layer.set_param(param["name"], param["value"])
            for i, shim_bytes in enumerate(data["shims"]):
                layer.shims[i].from_bytes(shim_bytes)
            for name, ref_i in data["refs"].items():
                if ref_i is None:
                    layer.set_ref(name, None)
                else:
                    layer.set_ref(name, nodes[ref_i])
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

    def __add__(self, other: Any) -> "Model":
        """Apply the function bound to the '+' operator."""
        if "+" not in self._thread_local.operators:
            raise TypeError("Undefined operator: +")
        return self._thread_local.operators["+"](self, other)

    def __sub__(self, other: Any) -> "Model":
        """Apply the function bound to the '-' operator."""
        if "-" not in self._thread_local.operators:
            raise TypeError("Undefined operator: -")
        return self._thread_local.operators["-"](self, other)

    def __mul__(self, other: Any) -> "Model":
        """Apply the function bound to the '*' operator."""
        if "*" not in self._thread_local.operators:
            raise TypeError("Undefined operator: *")
        return self._thread_local.operators["*"](self, other)

    def __matmul__(self, other: Any) -> "Model":
        """Apply the function bound to the '@' operator."""
        if "@" not in self._thread_local.operators:
            raise TypeError("Undefined operator: @")
        return self._thread_local.operators["@"](self, other)

    def __div__(self, other: Any) -> "Model":  # pragma: no cover
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __truediv__(self, other: Any) -> "Model":
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __floordiv__(self, other: Any) -> "Model":
        """Apply the function bound to the '//' operator."""
        if "//" not in self._thread_local.operators:
            raise TypeError("Undefined operator: //")
        return self._thread_local.operators["//"](self, other)

    def __mod__(self, other: Any) -> "Model":
        """Apply the function bound to the '%' operator."""
        if "%" not in self._thread_local.operators:
            raise TypeError("Undefined operator: %")
        return self._thread_local.operators["%"](self, other)

    def __pow__(self, other: Any, **kwargs) -> "Model":
        """Apply the function bound to the '**' operator."""
        if "**" not in self._thread_local.operators:
            raise TypeError("Undefined operator: **")
        return self._thread_local.operators["**"](self, other)

    def __lshift__(self, other: Any) -> "Model":
        """Apply the function bound to the '<<' operator."""
        if "<<" not in self._thread_local.operators:
            raise TypeError("Undefined operator: <<")
        return self._thread_local.operators["<<"](self, other)

    def __rshift__(self, other: Any) -> "Model":
        """Apply the function bound to the '>>' operator."""
        if ">>" not in self._thread_local.operators:
            raise TypeError("Undefined operator: >>")
        return self._thread_local.operators[">>"](self, other)

    def __and__(self, other: Any) -> "Model":
        """Apply the function bound to the '&' operator."""
        if "&" not in self._thread_local.operators:
            raise TypeError("Undefined operator: &")
        return self._thread_local.operators["&"](self, other)

    def __xor__(self, other: Any) -> "Model":
        """Apply the function bound to the '^' operator."""
        if "^" not in self._thread_local.operators:
            raise TypeError("Undefined operator: ^")
        return self._thread_local.operators["^"](self, other)

    def __or__(self, other: Any) -> "Model":
        """Apply the function bound to the '|' operator."""
        if "|" not in self._thread_local.operators:
            raise TypeError("Undefined operator: |")
        return self._thread_local.operators["|"](self, other)


__all__ = ["create_init", "Model"]
