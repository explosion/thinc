from typing import Any, Union, Sequence, cast, Dict, Optional, Callable, TypeVar, Type
import numpy
import threading
import random
import functools

try:  # pragma: no cover
    import cupy

    has_cupy = True
except (ImportError, AttributeError):
    cupy = None
    has_cupy = False

try:  # pragma: no cover
    import jax
    import jax.numpy

    has_jax = True
except ImportError:  # pragma: no cover
    jax = None
    has_jax = False

try:  # pragma: no cover
    import torch
    import torch.tensor
    import torch.utils.dlpack

    has_torch = True
except ImportError:  # pragma: no cover
    has_torch = False

try:  # pragma: no cover
    import tensorflow as tf

    has_tensorflow = True
except ImportError:  # pragma: no cover
    has_tensorflow = False

from .types import Array, Array2d, ArgsKwargs, Ragged, Padded


def get_array_module(arr):  # pragma: no cover
    if is_cupy_array(arr):
        return cupy
    elif is_jax_array(arr):
        return jax.numpy
    else:
        return numpy


def fix_random_seed(seed: int = 0) -> None:  # pragma: no cover
    """Set the random seed across random, numpy.random and cupy.random."""
    random.seed(seed)
    numpy.random.seed(seed)
    if cupy is not None:
        cupy.random.seed(seed)


def create_thread_local(
    attrs: Dict[str, Any], local_class: Type[threading.local] = threading.local
):
    obj = local_class()
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def is_xp_array(obj: Any) -> bool:
    """Check whether an object is a numpy or cupy array."""
    return is_numpy_array(obj) or is_cupy_array(obj) or is_jax_array(obj)


def is_cupy_array(obj: Any) -> bool:  # pragma: no cover
    """Check whether an object is a cupy array"""
    if not has_cupy:
        return False
    elif isinstance(obj, cupy.ndarray):
        return True
    else:
        return False


def is_jax_array(obj: Any) -> bool:  # pragma: no cover
    """Check whether an object is a jax.numpy array"""
    if not has_jax:
        return False
    elif isinstance(obj, numpy.ndarray):
        # Numpy arrays evaluate as True for instance of jax.numpy.ndarray :(
        return False
    elif isinstance(obj, jax.numpy.ndarray):
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


def to_numpy(data):  # pragma: no cover
    if isinstance(data, numpy.ndarray):
        return data
    elif has_cupy and isinstance(data, cupy.ndarray):
        return data.get()
    elif has_jax and isinstance(data, jax.numpy.ndarray):
        return jax.device_get(data)
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


def copy_array(dst: Array, src: Array) -> None:  # pragma: no cover
    if isinstance(dst, numpy.ndarray) and isinstance(src, numpy.ndarray):
        dst[:] = src
    elif is_cupy_array(dst):
        src = cupy.array(src, copy=False)
        cupy.copyto(dst, src)
    else:
        numpy.copyto(dst, src)


def to_categorical(Y: Array, n_classes: Optional[int] = None) -> Array2d:
    # From keras
    xp = get_array_module(Y)
    if xp is cupy:  # pragma: no cover
        Y = Y.get()
    Y = numpy.array(Y, dtype="int").ravel()
    if not n_classes:
        n_classes = numpy.max(Y) + 1
    n = Y.shape[0]
    categorical = numpy.zeros((n, n_classes), dtype="float32")
    categorical[numpy.arange(n), Y] = 1
    return xp.asarray(categorical)


def get_width(
    X: Union[Array, Ragged, Padded, Sequence[Array]], *, dim: int = -1
) -> int:
    """Infer the 'width' of a batch of data, which could be any of: Array,
    Ragged, Padded or Sequence of Arrays.
    """
    if isinstance(X, Ragged):
        return get_width(X.data, dim=dim)
    elif isinstance(X, Padded):
        return get_width(X.data, dim=dim)
    elif hasattr(X, "shape") and hasattr(X, "ndim"):
        X = cast(Array, X)
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


def xp2torch(
    xp_tensor: Array, requires_grad: bool = False
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


def torch2xp(torch_tensor: "torch.Tensor") -> Array:  # pragma: no cover
    """Convert a torch tensor to a numpy or cupy tensor."""
    if torch_tensor.is_cuda:
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    else:
        return torch_tensor.detach().numpy()


def xp2tensorflow(
    xp_tensor: Array, requires_grad: bool = False, as_variable: bool = False
) -> "tf.Tensor":  # pragma: no cover
    """Convert a numpy or cupy tensor to a TensorFlow Tensor or Variable"""
    assert_tensorflow_installed()
    tensorflow_tensor = tf.convert_to_tensor(xp_tensor)
    if as_variable:
        # tf.Variable() automatically puts in GPU if available.
        # So we need to control it using the context manager
        with tf.device(tensorflow_tensor.device):
            tensorflow_tensor = tf.Variable(tensorflow_tensor, trainable=requires_grad)
    if requires_grad is False and as_variable is False:
        # tf.stop_gradient() automatically puts in GPU if available.
        # So we need to control it using the context manager
        with tf.device(tensorflow_tensor.device):
            tensorflow_tensor = tf.stop_gradient(tensorflow_tensor)
    return tensorflow_tensor


def tensorflow2xp(tensorflow_tensor: "tf.Tensor") -> Array:  # pragma: no cover
    """Convert a Tensorflow tensor to numpy or cupy tensor."""
    assert_tensorflow_installed()
    return tensorflow_tensor.numpy()


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


__all__ = [
    "get_array_module",
    "fix_random_seed",
    "create_thread_local",
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
]
