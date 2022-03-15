from typing import Any, Union, Sequence, cast, Dict, Optional, Callable, TypeVar
from typing import List, Tuple
import numpy
from packaging.version import Version
import random
import functools
from wasabi import table
from pydantic import create_model, ValidationError
import inspect
import os
import tempfile
import threading
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass

DATA_VALIDATION: ContextVar[bool] = ContextVar("DATA_VALIDATION", default=False)

try:  # pragma: no cover
    import cupy

    has_cupy = True
except (ImportError, AttributeError):
    cupy = None
    has_cupy = False


try:  # pragma: no cover
    import torch
    from torch import tensor
    import torch.utils.dlpack

    has_torch = True
    has_torch_gpu = torch.cuda.device_count() != 0
    torch_version = Version(str(torch.__version__))
    has_torch_amp = (
        torch_version >= Version("1.9.0")
        and not torch.cuda.amp.common.amp_definitely_not_available()
    )
except ImportError:  # pragma: no cover
    has_torch = False
    has_torch_gpu = False
    has_torch_amp = False
    torch_version = Version("0.0.0")

try:  # pragma: no cover
    import tensorflow.experimental.dlpack
    import tensorflow as tf

    has_tensorflow = True
except ImportError:  # pragma: no cover
    has_tensorflow = False


try:  # pragma: no cover
    import mxnet as mx

    has_mxnet = True
except ImportError:  # pragma: no cover
    has_mxnet = False

from .types import ArrayXd, ArgsKwargs, Ragged, Padded, FloatsXd, IntsXd  # noqa: E402
from . import types  # noqa: E402


def get_array_module(arr):  # pragma: no cover
    if is_cupy_array(arr):
        return cupy
    else:
        return numpy


def gpu_is_available():
    try:
        cupy.cuda.runtime.getDeviceCount()
        return True
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


def fix_random_seed(seed: int = 0) -> None:  # pragma: no cover
    """Set the random seed across random, numpy.random and cupy.random."""
    random.seed(seed)
    numpy.random.seed(seed)
    if has_torch:
        torch.manual_seed(seed)
    if has_cupy and gpu_is_available():
        cupy.random.seed(seed)
        if has_torch and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def is_xp_array(obj: Any) -> bool:
    """Check whether an object is a numpy or cupy array."""
    return is_numpy_array(obj) or is_cupy_array(obj)


def is_cupy_array(obj: Any) -> bool:  # pragma: no cover
    """Check whether an object is a cupy array"""
    if not has_cupy:
        return False
    elif isinstance(obj, cupy.ndarray):
        return True
    else:
        return False


def is_numpy_array(obj: Any) -> bool:
    """Check whether an object is a numpy array"""
    if isinstance(obj, numpy.ndarray):
        return True
    else:
        return False


def is_torch_array(obj: Any) -> bool:  # pragma: no cover
    if torch is None:
        return False
    elif isinstance(obj, torch.Tensor):
        return True
    else:
        return False


def is_tensorflow_array(obj: Any) -> bool:  # pragma: no cover
    if not has_tensorflow:
        return False
    elif isinstance(obj, tf.Tensor):
        return True
    else:
        return False


def is_mxnet_array(obj: Any) -> bool:  # pragma: no cover
    if not has_mxnet:
        return False
    elif isinstance(obj, mx.nd.NDArray):
        return True
    else:
        return False


def to_numpy(data):  # pragma: no cover
    if isinstance(data, numpy.ndarray):
        return data
    elif has_cupy and isinstance(data, cupy.ndarray):
        return data.get()
    else:
        return numpy.array(data)


def set_active_gpu(gpu_id: int) -> "cupy.cuda.Device":  # pragma: no cover
    """Set the current GPU device for cupy and torch (if available)."""
    import cupy.cuda.device

    device = cupy.cuda.device.Device(gpu_id)
    device.use()
    try:
        import torch

        torch.cuda.set_device(gpu_id)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    except ImportError:
        pass
    return device


def require_cpu() -> bool:  # pragma: no cover
    """Use CPU through best available backend."""
    from .backends import set_current_ops, get_ops

    ops = get_ops("cpu")
    set_current_ops(ops)
    set_torch_tensor_type_for_ops(ops)

    return True


def prefer_gpu(gpu_id: int = 0) -> bool:  # pragma: no cover
    """Use GPU if it's available. Returns True if so, False otherwise."""
    from .backends.cupy_ops import CupyOps

    if CupyOps.xp is None:
        return False
    else:
        require_gpu(gpu_id=gpu_id)
        return True


def require_gpu(gpu_id: int = 0) -> bool:  # pragma: no cover
    from .backends import set_current_ops, CupyOps

    if CupyOps.xp is None:
        raise ValueError("GPU is not accessible. Was the library installed correctly?")

    set_current_ops(CupyOps())
    set_active_gpu(gpu_id)
    return True


def copy_array(dst: ArrayXd, src: ArrayXd) -> None:  # pragma: no cover
    if isinstance(dst, numpy.ndarray) and isinstance(src, numpy.ndarray):
        dst[:] = src
    elif is_cupy_array(dst):
        src = cupy.array(src, copy=False)
        cupy.copyto(dst, src)
    else:
        numpy.copyto(dst, src)  # type: ignore


def to_categorical(
    Y: IntsXd,
    n_classes: Optional[int] = None,
    *,
    label_smoothing: float = 0.0,
) -> FloatsXd:
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(
            "label_smoothing should be greater or "
            "equal to 0.0 and less than 0.5, "
            f"but {label_smoothing} was provided."
        )

    if n_classes is None:
        n_classes = int(numpy.max(Y) + 1)  # type: ignore

    if label_smoothing == 0.0:
        if n_classes == 0:
            raise ValueError("n_classes should be at least 1")
        nongold_prob = 0.0
    else:
        if not n_classes > 1:
            raise ValueError(
                "n_classes should be greater than 1 when label smoothing is enabled,"
                f"but {n_classes} was provided."
            )
        nongold_prob = label_smoothing / (n_classes - 1)

    xp = get_array_module(Y)
    label_distr = xp.full((n_classes, n_classes), nongold_prob, dtype="float32")
    xp.fill_diagonal(label_distr, 1 - label_smoothing)
    return label_distr[Y]


def get_width(
    X: Union[ArrayXd, Ragged, Padded, Sequence[ArrayXd]], *, dim: int = -1
) -> int:
    """Infer the 'width' of a batch of data, which could be any of: Array,
    Ragged, Padded or Sequence of Arrays.
    """
    if isinstance(X, Ragged):
        return get_width(X.data, dim=dim)
    elif isinstance(X, Padded):
        return get_width(X.data, dim=dim)
    elif hasattr(X, "shape") and hasattr(X, "ndim"):
        X = cast(ArrayXd, X)
        if len(X.shape) == 0:
            return 0
        elif len(X.shape) == 1:
            return int(X.max()) + 1
        else:
            return X.shape[dim]
    elif isinstance(X, (list, tuple)):
        if len(X) == 0:
            return 0
        else:
            return get_width(X[0], dim=dim)
    else:
        err = "Cannot get width of object: has neither shape nor __getitem__"
        raise ValueError(err)


def assert_tensorflow_installed() -> None:  # pragma: no cover
    """Raise an ImportError if TensorFlow is not installed."""
    template = "TensorFlow support requires {pkg}: pip install thinc[tensorflow]"
    if not has_tensorflow:
        raise ImportError(template.format(pkg="tensorflow>=2.0.0"))


def assert_mxnet_installed() -> None:  # pragma: no cover
    """Raise an ImportError if MXNet is not installed."""
    if not has_mxnet:
        raise ImportError("MXNet support requires mxnet: pip install thinc[mxnet]")


def assert_pytorch_installed() -> None:  # pragma: no cover
    """Raise an ImportError if PyTorch is not installed."""
    if not has_torch:
        raise ImportError("PyTorch support requires torch: pip install thinc[torch]")


def convert_recursive(
    is_match: Callable[[Any], bool], convert_item: Callable[[Any], Any], obj: Any
) -> Any:
    """Either convert a single value if it matches a given function, or
    recursively walk over potentially nested lists, tuples and dicts applying
    the conversion, and returns the same type. Also supports the ArgsKwargs
    dataclass.
    """
    if is_match(obj):
        return convert_item(obj)
    elif isinstance(obj, ArgsKwargs):
        converted = convert_recursive(is_match, convert_item, list(obj.items()))
        return ArgsKwargs.from_items(converted)
    elif isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            key = convert_recursive(is_match, convert_item, key)
            value = convert_recursive(is_match, convert_item, value)
            converted[key] = value
        return converted
    elif isinstance(obj, list):
        return [convert_recursive(is_match, convert_item, item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_recursive(is_match, convert_item, item) for item in obj)
    else:
        return obj


def iterate_recursive(is_match: Callable[[Any], bool], obj: Any) -> Any:
    """Either yield a single value if it matches a given function, or recursively
    walk over potentially nested lists, tuples and dicts yielding matching
    values. Also supports the ArgsKwargs dataclass.
    """
    if is_match(obj):
        yield obj
    elif isinstance(obj, ArgsKwargs):
        yield from iterate_recursive(is_match, list(obj.items()))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iterate_recursive(is_match, key)
            yield from iterate_recursive(is_match, value)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            yield from iterate_recursive(is_match, item)


def xp2torch(
    xp_tensor: ArrayXd, requires_grad: bool = False
) -> "torch.Tensor":  # pragma: no cover
    """Convert a numpy or cupy tensor to a PyTorch tensor."""
    if hasattr(xp_tensor, "toDlpack"):
        dlpack_tensor = xp_tensor.toDlpack()  # type: ignore
        torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
    else:
        torch_tensor = torch.from_numpy(xp_tensor)
    if requires_grad:
        torch_tensor.requires_grad_()
    return torch_tensor


def torch2xp(torch_tensor: "torch.Tensor") -> ArrayXd:  # pragma: no cover
    """Convert a torch tensor to a numpy or cupy tensor."""
    if torch_tensor.is_cuda:
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    else:
        return torch_tensor.detach().numpy()


def xp2tensorflow(
    xp_tensor: ArrayXd, requires_grad: bool = False, as_variable: bool = False
) -> "tf.Tensor":  # pragma: no cover
    """Convert a numpy or cupy tensor to a TensorFlow Tensor or Variable"""
    assert_tensorflow_installed()
    if hasattr(xp_tensor, "toDlpack"):
        dlpack_tensor = xp_tensor.toDlpack()  # type: ignore
        tf_tensor = tensorflow.experimental.dlpack.from_dlpack(dlpack_tensor)
    else:
        tf_tensor = tf.convert_to_tensor(xp_tensor)
    if as_variable:
        # tf.Variable() automatically puts in GPU if available.
        # So we need to control it using the context manager
        with tf.device(tf_tensor.device):
            tf_tensor = tf.Variable(tf_tensor, trainable=requires_grad)
    if requires_grad is False and as_variable is False:
        # tf.stop_gradient() automatically puts in GPU if available.
        # So we need to control it using the context manager
        with tf.device(tf_tensor.device):
            tf_tensor = tf.stop_gradient(tf_tensor)
    return tf_tensor


def tensorflow2xp(tf_tensor: "tf.Tensor") -> ArrayXd:  # pragma: no cover
    """Convert a Tensorflow tensor to numpy or cupy tensor."""
    assert_tensorflow_installed()
    if tf_tensor.device is not None:
        _, device_type, device_num = tf_tensor.device.rsplit(":", 2)
    else:
        device_type = "CPU"
    if device_type == "CPU" or not has_cupy:
        return tf_tensor.numpy()
    else:
        dlpack_tensor = tensorflow.experimental.dlpack.to_dlpack(tf_tensor)
        return cupy.fromDlpack(dlpack_tensor)


def xp2mxnet(
    xp_tensor: ArrayXd, requires_grad: bool = False
) -> "mx.nd.NDArray":  # pragma: no cover
    """Convert a numpy or cupy tensor to a MXNet tensor."""
    if hasattr(xp_tensor, "toDlpack"):
        dlpack_tensor = xp_tensor.toDlpack()  # type: ignore
        mx_tensor = mx.nd.from_dlpack(dlpack_tensor)
    else:
        mx_tensor = mx.nd.from_numpy(xp_tensor)
    if requires_grad:
        mx_tensor.attach_grad()
    return mx_tensor


def mxnet2xp(mx_tensor: "mx.nd.NDArray") -> ArrayXd:  # pragma: no cover
    """Convert a MXNet tensor to a numpy or cupy tensor."""
    if mx_tensor.context.device_type != "cpu":
        return cupy.fromDlpack(mx_tensor.to_dlpack_for_write())
    else:
        return mx_tensor.detach().asnumpy()


# This is how functools.partials seems to do it, too, to retain the return type
PartialT = TypeVar("PartialT")


def partial(
    func: Callable[..., PartialT], *args: Any, **kwargs: Any
) -> Callable[..., PartialT]:
    """Wrapper around functools.partial that retains docstrings and can include
    other workarounds if needed.
    """
    partial_func = functools.partial(func, *args, **kwargs)
    partial_func.__doc__ = func.__doc__
    return partial_func


class DataValidationError(ValueError):
    def __init__(
        self, name: str, X: Any, Y: Any, errors: List[Dict[str, Any]] = []
    ) -> None:
        """Custom error for validating inputs / outputs at runtime."""
        message = f"Data validation error in '{name}'"
        type_info = f"X: {type(X)} Y: {type(Y)}"
        data = []
        for error in errors:
            err_loc = " -> ".join([str(p) for p in error.get("loc", [])])
            data.append((err_loc, error.get("msg")))
        result = [message, type_info, table(data)]
        ValueError.__init__(self, "\n\n" + "\n".join(result))


class _ArgModelConfig:
    extra = "forbid"
    arbitrary_types_allowed = True


def validate_fwd_input_output(
    name: str, func: Callable[[Any, Any, bool], Any], X: Any, Y: Any
) -> None:
    """Validate the input and output of a forward function against the type
    annotations, if available. Used in Model.initialize with the input and
    output samples as they pass through the network.
    """
    sig = inspect.signature(func)
    empty = inspect.Signature.empty
    params = list(sig.parameters.values())
    if len(params) != 3:
        bad_params = f"{len(params)} ({', '.join([p.name for p in params])})"
        err = f"Invalid forward function. Expected 3 arguments (model, X , is_train), got {bad_params}"
        raise DataValidationError(name, X, Y, [{"msg": err}])
    annot_x = params[1].annotation
    annot_y = sig.return_annotation
    sig_args: Dict[str, Any] = {"__config__": _ArgModelConfig}
    args = {}
    if X is not None and annot_x != empty:
        if isinstance(X, list) and len(X) > 5:
            X = X[:5]
        sig_args["X"] = (annot_x, ...)
        args["X"] = X
    if Y is not None and annot_y != empty:
        if isinstance(Y, list) and len(Y) > 5:
            Y = Y[:5]
        sig_args["Y"] = (annot_y, ...)
        args["Y"] = (Y, lambda x: x)
    ArgModel = create_model("ArgModel", **sig_args)
    # Make sure the forward refs are resolved and the types used by them are
    # available in the correct scope. See #494 for details.
    ArgModel.update_forward_refs(**types.__dict__)
    try:
        ArgModel.parse_obj(args)
    except ValidationError as e:
        raise DataValidationError(name, X, Y, e.errors()) from None


@contextlib.contextmanager
def make_tempfile(mode="r"):
    f = tempfile.NamedTemporaryFile(mode=mode, delete=False)
    yield f
    f.close()
    os.remove(f.name)


@contextlib.contextmanager
def data_validation(validation):
    with threading.Lock():
        prev = DATA_VALIDATION.get()
        DATA_VALIDATION.set(validation)
        yield
        DATA_VALIDATION.set(prev)


@contextlib.contextmanager
def use_nvtx_range(message: int, id_color: int = -1):
    """Context manager to register the executed code as an NVTX range. The
    ranges can be used as markers in CUDA profiling."""
    if has_cupy:
        cupy.cuda.nvtx.RangePush(message, id_color)
        yield
        cupy.cuda.nvtx.RangePop()
    else:
        yield


def set_torch_tensor_type_for_ops(ops):
    """Set the PyTorch default tensor type for the given ops. This is a
    no-op if PyTorch is not available."""
    from .backends.cupy_ops import CupyOps

    try:
        import torch

        if CupyOps.xp is not None and isinstance(ops, CupyOps):
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
    except ImportError:
        pass


@dataclass
class ArrayInfo:
    """Container for info for checking array compatibility."""

    shape: types.Shape
    dtype: types.DTypes

    @classmethod
    def from_array(cls, arr: ArrayXd):
        return cls(shape=arr.shape, dtype=arr.dtype)

    def check_consistency(self, arr: ArrayXd):
        if arr.shape != self.shape:
            raise ValueError(
                f"Shape mismatch in backprop. Y: {self.shape}, dY: {arr.shape}"
            )
        if arr.dtype != self.dtype:
            raise ValueError(
                f"Type mismatch in backprop. Y: {self.dtype}, dY: {arr.dtype}"
            )


__all__ = [
    "get_array_module",
    "fix_random_seed",
    "is_cupy_array",
    "is_numpy_array",
    "set_active_gpu",
    "prefer_gpu",
    "require_gpu",
    "copy_array",
    "to_categorical",
    "get_width",
    "xp2torch",
    "torch2xp",
    "tensorflow2xp",
    "xp2tensorflow",
    "validate_fwd_input_output",
    "DataValidationError",
    "make_tempfile",
    "use_nvtx_range",
    "set_torch_tensor_type_for_ops",
    "ArrayInfo",
]
