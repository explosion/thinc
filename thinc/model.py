from typing import Dict, List, Callable, Optional, Any, Union, Iterable, Set
from typing import Generic, Sequence, Tuple, TypeVar
import contextlib
import srsly
from pathlib import Path
import copy
import functools

from .backends import ParamServer, Ops, NumpyOps, CupyOps, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .shims import Shim
from .util import get_width, create_thread_local
from .types import Array


InT = TypeVar("InT")
OutT = TypeVar("OutT")


class create_init:
    """Create an init function, given a dictionary of parameter initializers."""

    def __init__(self, initializers: Dict[str, Callable]):
        self.initializers = initializers

    def __call__(self, model, X: Optional[Array] = None, Y: Optional[Array] = None
             ) -> None:
        if X is not None:
            model.set_dim("nI", get_width(X))
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        W = model.ops.alloc_f2d(model.get_dim("nO"), model.get_dim("nI"))
        b = model.ops.alloc_f1d(model.get_dim("nO"))
        if "W" in self.initializers:
            self.initializers["W"](W, inplace=True)
        if "b" in self.initializers:
            self.initializers["b"](b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)


def empty_init(*a, **k):
    return None


class Model(Generic[InT, OutT]):
    """Class for implementing Thinc models and layers."""

    global_id: int = 0
    _thread_local = create_thread_local({"operators": {}})

    name: str
    ops: Union[NumpyOps, CupyOps]  # TODO: This is wrong, should be Ops
    id: int
    _func: Callable
    _init: Callable
    _params: ParamServer
    _dims: Dict[str, Optional[int]]
    _layers: List["Model"]
    _shims: List[Shim]
    _attrs: Dict[str, Any]
    _has_params: Dict[str, Optional[bool]]

    # This "locks" the class, so we get an error if you try to assign to
    # an unexpected variable.
    __slots__ = [
        "name",
        "id",
        "ops",
        "_func",
        "_init",
        "_params",
        "_dims",
        "_attrs",
        "_refs",
        "_layers",
        "_shims",
        "_has_params",
    ]

    def __init__(
        self,
        name: str,
        forward: Callable,
        *,
        init: Callable = empty_init,
        dims: Dict[str, Optional[int]] = {},
        params: Dict[str, Optional[Array]] = {},
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
        self._params = ParamServer()
        self._dims = dict(dims)
        self._attrs = dict(attrs)
        self._refs = dict(refs)
        self._layers = list(layers)
        self._shims = list(shims)
        # Take care to increment the base class here! It needs to be unique
        # across all models.
        Model.global_id += 1
        self.id = Model.global_id
        self._has_params = {}
        for name, value in params.items():
            self._has_params[name] = None
            if value is not None:
                self.set_param(name, value)

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
        return tuple(self._has_params.keys())

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
        if name not in self._has_params:
            return False
        elif self._has_params[name] is not None:
            return True
        else:
            return None

    def get_param(self, name: str) -> Array:
        """Retrieve a weights parameter by name."""
        if name not in self._has_params:
            raise KeyError(f"Unknown param: '{name}' for model '{self.name}'.")
        if not self._params.has_param(self.id, name):
            raise KeyError(
                f"Parameter '{name}' for model '{self.name}' has not been allocated yet."
            )
        return self._params.get_param(self.id, name)

    def set_param(self, name: str, value: Optional[Array]) -> None:
        """Set a weights parameter's value."""
        if value is None:
            self._has_params[name] = None
        else:
            self._params.set_param(self.id, name, value)
            self._has_params[name] = True

    def has_grad(self, name: str) -> bool:
        """Check whether the model has a non-zero gradient for a parameter.
        """
        return self._params.has_grad(self.id, name)

    def get_grad(self, name: str) -> Array:
        """Get a gradient from the model."""
        return self._params.get_grad(self.id, name)

    def set_grad(self, name: str, value: Array) -> None:
        """Set a gradient value for the model."""
        self._params.set_grad(self.id, name, value)

    def inc_grad(self, name: str, value: Array) -> None:
        """Check whether the model has a gradient of the given name."""
        self._params.inc_grad(self.id, name, value)

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
        for node in self.walk():
            for name in node.param_names:
                if node.has_grad(name):
                    param = node.get_param(name)
                    grad = node.get_grad(name)
                    param, grad = optimizer((node.id, name), param, grad)
                    node.set_param(name, param)
                    node.set_grad(name, grad)
            for shim in node.shims:
                shim.finish_update(optimizer)

    @contextlib.contextmanager
    def use_params(self, params: Dict[Tuple[int, str], Array]):
        """Context manager to temporarily set the model's parameters to
        specified values. The params are a dictionary keyed by model IDs, whose
        values are arrays of weight values.
        """
        backup = {}
        for name in self.param_names:
            key = (self.id, name)
            if key in params:
                backup[name] = self.get_param(name)
                self.set_param(name, params[key])

        with contextlib.ExitStack() as stack:
            for layer in self.layers:
                stack.enter_context(layer.use_params(params))
            for shim in self.shims:
                stack.enter_context(shim.use_params(params))
            yield
        if backup:
            for name, param in backup.items():
                self.set_param(name, param)

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

    def get_gradients(self) -> Dict[Tuple[int, str], Tuple[Array, Array]]:
        """Get non-zero gradients of the model's parameters, as a dictionary
        keyed by the parameter ID. The values are (weights, gradients) tuples.
        """
        gradients = {}
        for node in self.walk():
            for name in node.grad_names:
                param = node.get_param(name)
                grad = node.get_grad(name)
                gradients[(node.id, name)] = (param, grad)
        return gradients

    def copy(self) -> "Model":
        """
        Create a copy of the model, its attributes, and its parameters. Any child
        layers will also be deep-copied. The copy will receive a distinct `model.id`
        value.
        """
        params = {}
        grads = {}
        for name in self.param_names:
            params[name] = self.get_param(name) if self.has_param(name) else None
        for name in self.grad_names:
            grads[name] = self.get_grad(name)

        copied: Model[InT, OutT] = Model(
            self.name,
            self._func,
            init=self._init,
            params=copy.deepcopy(params),
            dims=copy.deepcopy(self._dims),
            attrs=copy.deepcopy(self._attrs),
            layers=[layer.copy() for layer in self.layers],
            shims=[shim.copy() for shim in self.shims],
        )
        for name in self.grad_names:
            copied.set_grad(name, self.get_grad(name).copy())
        return copied

    def to_gpu(self, gpu_id: int) -> None:  # pragma: no cover
        """Transfer the model to a given GPU device."""
        import cupy.cuda.device

        device = cupy.cuda.device.Device(gpu_id)
        with device.use():
            self._to_ops(CupyOps())

    def to_cpu(self) -> None:  # pragma: no cover
        """Transfer the model to CPU."""
        self._to_ops(NumpyOps())

    def _to_ops(self, ops: Ops) -> None:  # pragma: no cover
        """Common method for to_cpu/to_gpu."""
        for node in self.walk():
            node.ops = ops
            for name in node.param_names:
                if node.has_param(name):
                    node.set_param(name, ops.asarray(node.get_param(name)))
                if node.has_grad(name):
                    node.set_grad(name, ops.asarray(node.get_grad(name)))
            for shim in node.shims:
                shim.to_device(ops.device)

    def to_bytes(self) -> bytes:
        """Serialize the model to a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        # We separate out like this to make it easier to read the data in chunks.
        # The shims might have large weights, while the nodes data will be
        # small. The attrs are probably not very large, but could be.
        # The lists are aligned, and refer to the order of self.walk().
        msg: Dict[str, List] = {"nodes": [], "attrs": [], "params": [], "shims": []}
        nodes = list(self.walk())
        # Serialize references by their index into the flattened tree.
        # This is the main reason we can't accept out-of-tree references:
        # we'd have no way to serialize/deserialize them.
        node_to_i: Dict[int, Optional[int]]
        node_to_i = {node.id: i for i, node in enumerate(nodes)}
        for i, node in enumerate(nodes):
            refs: Dict[str, Optional[int]] = {}
            invalid_refs: List[str] = []
            for name in node.ref_names:
                if not node.has_ref(name):
                    refs[name] = None
                else:
                    ref = node.get_ref(name)
                    if ref.id in node_to_i:
                        refs[name] = node_to_i[ref.id]
                    else:
                        invalid_refs.append(name)
            if invalid_refs:
                raise ValueError(f"Cannot get references: {invalid_refs}")
            dims = {}
            for dim in node.dim_names:
                dims[dim] = node.get_dim(dim) if node.has_dim(dim) else None
            msg["nodes"].append(
                {
                    "index": i,
                    "name": node.name,
                    "dims": dims,
                    "refs": refs,
                }
            )
        for node in nodes:
            attrs = {}
            for name in node.attr_names:
                value = node.get_attr(name)
                try:
                    attrs[name] = serialize_attr(value, value, name, node)
                except TypeError:
                    continue
            msg["attrs"].append(attrs)
        for node in nodes:
            msg["shims"].append([shim.to_bytes() for shim in node.shims])
        for node in nodes:
            params: Dict[str, Optional[Array]] = {}
            for name in node.param_names:
                if node.has_param(name):
                    params[name] = node.get_param(name)
                else:
                    params[name] = None
            msg["params"].append(params)
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data: bytes) -> "Model":
        """Deserialize the model from a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        msg = srsly.msgpack_loads(bytes_data)
        nodes = list(self.walk())
        if len(msg["nodes"]) != len(nodes):
            raise ValueError("Cannot deserialize model: mismatched structure.")
        for i, node in enumerate(nodes):
            info = msg["nodes"][i]
            node.name = info["name"]
            for dim, value in info["dims"].items():
                if value is not None:
                    node.set_dim(dim, value)
            for ref, ref_index in info["refs"].items():
                if ref_index is None:
                    node.set_ref(ref, None)
                else:
                    node.set_ref(ref, nodes[ref_index])
            for attr, value in msg["attrs"][i].items():
                default_value = node.get_attr(attr)
                loaded_value = deserialize_attr(default_value, value, attr, node)
                node.set_attr(attr, loaded_value)
            for param_name, value in msg["params"][i].items():
                node.set_param(param_name, value)
            for i, shim_bytes in enumerate(msg["shims"][i]):
                node.shims[i].from_bytes(shim_bytes)
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


@functools.singledispatch
def serialize_attr(_: Any, value: Any, name: str, model: Model) -> bytes:
    """Serialize an attribute value (defaults to msgpack). You can register
    custom serializers using the @serialize_attr.register decorator with the
    type to serialize, e.g.: @serialize_attr.register(MyCustomObject).
    """
    return srsly.msgpack_dumps(value)


@functools.singledispatch
def deserialize_attr(_: Any, value: Any, name: str, model: Model) -> Any:
    """Deserialize an attribute value (defaults to msgpack). You can register
    custom deserializers using the @deserialize_attr.register decorator with the
    type to deserialize, e.g.: @deserialize_attr.register(MyCustomObject).
    """
    return srsly.msgpack_loads(value)


_ModelT = TypeVar("_ModelT", bound=Model)


def change_attr_values(model: _ModelT, mapping: Dict[str, Dict[str, Any]]) -> _ModelT:
    """Walk over the model's nodes, changing the value of attributes using the
    provided mapping, which maps node names to attr names to attr values.
    """
    for node in model.walk():
        if node.name in mapping:
            attrs = mapping[node.name]
            for attr, value in attrs.items():
                if node.has_attr(attr):
                    node.set_attr(attr, value)
    return model


def set_dropout_rate(model: _ModelT, drop: float, attrs={"dropout": "rate"}) -> _ModelT:
    """Walk over the model's nodes, setting the dropout rate. Dropout nodes are
    identified by name. You can configure the name-to-attribute mapping using
    the `attrs` dict.
    """
    mapping = {name: {attr: drop} for name, attr in attrs.items()}
    return change_attr_values(model, mapping)


__all__ = [
    "create_init",
    "Model",
    "serialize_attr",
    "deserialize_attr",
    "change_attr_values",
    "set_dropout_rate",
]
