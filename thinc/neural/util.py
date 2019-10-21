# coding: utf8
from __future__ import print_function, unicode_literals

import numpy
from pathlib import Path
import itertools


try:
    import cupy
    from cupy import get_array_module
except ImportError:
    cupy = None
    get_array_module = lambda _: numpy

try:
    basestring
except NameError:
    basestring = str

try:
    unicode
except NameError:
    unicode = str


def is_cupy_array(arr):
    """Check whether an array is a cupy array"""
    if cupy is None:
        return False
    elif isinstance(arr, cupy.ndarray):
        return True
    else:
        return False


def is_numpy_array(arr):
    """Check whether an array is a numpy array"""
    if isinstance(arr, numpy.ndarray):
        return True
    else:
        return False


def get_ops(ops):
    from .ops import NumpyOps, CupyOps

    if ops in ("numpy", "cpu") or (isinstance(ops, int) and ops < 0):
        return NumpyOps
    elif ops in ("cupy", "gpu") or (isinstance(ops, int) and ops >= 0):
        return CupyOps
    else:
        raise ValueError("Invalid ops (or device) description: %s" % ops)


def set_active_gpu(gpu_id):
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


def prefer_gpu(gpu_id=0):
    """Use GPU if it's available. Returns True if so, False otherwise."""
    from .ops import CupyOps

    if CupyOps.xp is None:
        return False
    else:
        require_gpu(gpu_id=gpu_id)
        return True


def require_gpu(gpu_id=0):
    from ._classes.model import Model
    from .ops import CupyOps

    if CupyOps.xp is None:
        raise ValueError("GPU is not accessible. Was the library installed correctly?")
    Model.Ops = CupyOps
    Model.ops = CupyOps()
    set_active_gpu(gpu_id)
    return True


def minibatch(items, size=8):
    """Iterate over batches of items. `size` may be an iterator,
    so that batch-size can vary on each step.
    """
    if isinstance(size, int):
        size_ = itertools.repeat(size)
    else:
        size_ = size
    if hasattr(items, "__len__") and hasattr(items, "__getitem__"):
        i = 0
        while i < len(items):
            batch_size = next(size_)
            yield items[i : i + batch_size]
            i += batch_size
    else:
        items = iter(items)
        while True:
            batch_size = next(size_)
            batch = list(itertools.islice(items, int(batch_size)))
            if len(batch) == 0:
                break
            yield list(batch)


def mark_sentence_boundaries(sequences, drop=0.0):  # pragma: no cover
    """Pad sentence sequences with EOL markers."""
    for sequence in sequences:
        sequence.insert(0, "-EOL-")
        sequence.insert(0, "-EOL-")
        sequence.append("-EOL-")
        sequence.append("-EOL-")
    return sequences, None


def remap_ids(ops):
    id_map = {0: 0}

    def begin_update(ids, drop=0.0):
        n_vector = len(id_map)
        for i, id_ in enumerate(ids):
            if id_ not in id_map:
                id_map[id_] = n_vector
                n_vector += 1
            ids[i] = id_map[id_]
        return ids, None

    return begin_update


def copy_array(dst, src, casting="same_kind", where=None):
    if isinstance(dst, numpy.ndarray) and isinstance(src, numpy.ndarray):
        dst[:] = src
    elif is_cupy_array(dst):
        src = cupy.array(src, copy=False)
        cupy.copyto(dst, src)
    else:
        numpy.copyto(dst, src)


def ensure_path(path):
    if isinstance(path, unicode) or isinstance(path, str):
        return Path(path)
    else:
        return path


def to_categorical(y, nb_classes=None):
    # From keras
    xp = get_array_module(y)
    if xp is cupy:
        y = y.get()
    y = numpy.array(y, dtype="int").ravel()
    if not nb_classes:
        nb_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, nb_classes), dtype="float32")
    categorical[numpy.arange(n), y] = 1
    return xp.asarray(categorical)


def flatten_sequences(sequences, drop=0.0):  # pragma: no cover
    xp = get_array_module(sequences[0])
    return xp.concatenate(sequences), None


def partition(examples, split_size):  # pragma: no cover
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]
