from typing import Iterable, Any, Union, Tuple, Iterator, Sequence, cast, Dict
from typing import Optional, Callable, TypeVar
import numpy
import itertools
import threading
import random
import functools

try:  # pragma: no cover
    import cupy
    from cupy import get_array_module
except (ImportError, AttributeError):
    cupy = None
    get_array_module = lambda _: numpy

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

from .types import Array, Ragged, Padded, ArgsKwargs, RNNState, Array2d


def fix_random_seed(seed: int = 0) -> None:  # pragma: no cover
    """Set the random seed across random, numpy.random and cupy.random."""
    random.seed(seed)
    numpy.random.seed(seed)
    if cupy is not None:
        cupy.random.seed(seed)


def create_thread_local(attrs: Dict[str, Any]):
    obj = threading.local()
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def is_xp_array(obj: Any) -> bool:
    """Check whether an object is a numpy or cupy array."""
    return is_numpy_array(obj) or is_cupy_array(obj)


def is_cupy_array(obj: Any) -> bool:  # pragma: no cover
    """Check whether an object is a cupy array"""
    if cupy is None:
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


def get_shuffled_batches(
    X: Array, Y: Array, batch_size
) -> Iterable[Tuple[Array, Array]]:
    """Iterate over paired batches from two arrays, shuffling the indices."""
    xp = get_array_module(X)
    indices = xp.arange(X.shape[0], dtype="i")
    xp.random.shuffle(indices)
    for index_batch in minibatch(indices, size=batch_size):
        yield X[index_batch], Y[index_batch]


def minibatch(
    items: Iterable[Any], size: Union[int, Iterator[int]] = 8
) -> Iterable[Any]:
    """Iterate over batches of items. `size` may be an iterator,
    so that batch-size can vary on each step.
    """
    if isinstance(size, int):
        size_ = itertools.repeat(size)
    else:
        size_ = size
    if hasattr(items, "__len__") and hasattr(items, "__getitem__"):
        i = 0
        while i < len(items):  # type: ignore
            batch_size = next(size_)
            yield items[i : i + batch_size]  # type: ignore
            i += batch_size
    else:
        items = iter(items)
        while True:
            batch_size = next(size_)
            batch = list(itertools.islice(items, int(batch_size)))
            if len(batch) == 0:
                break
            yield list(batch)


def evaluate_model_on_arrays(
    model, inputs: Array, labels: Array, batch_size: int
) -> float:
    """Helper to evaluate accuracy of a model in the simplest cases, where
    there's one correct output class and the inputs are arrays. Not guaranteed
    to cover all situations – many applications will have to implement their
    own evaluation methods.
    """
    score = 0.0
    total = 0.0
    for i in range(0, inputs.shape[0], batch_size):
        X = inputs[i : i + batch_size]
        Y = labels[i : i + batch_size]
        Yh = model.predict(X)
        score += (Y.argmax(axis=1) == Yh.argmax(axis=1)).sum()
        total += Yh.shape[0]
    return score / total


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
    X: Union[Array, Ragged, Padded, Sequence[Array], RNNState], *, dim: int = -1
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
    "fix_random_seed",
    "create_thread_local",
    "is_cupy_array",
    "is_numpy_array",
    "set_active_gpu",
    "prefer_gpu",
    "require_gpu",
    "copy_array",
    "get_shuffled_batches",
    "minibatch",
    "evaluate_model_on_arrays",
    "to_categorical",
    "get_width",
    "xp2torch",
    "torch2xp",
    "tensorflow2xp",
    "xp2tensorflow",
]
