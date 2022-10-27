from .config import Config, registry, ConfigValidationError
from .initializers import normal_init, uniform_init, glorot_uniform_init, zero_init
from .initializers import configure_normal_init
from .loss import CategoricalCrossentropy, L2Distance, CosineDistance
from .loss import SequenceCategoricalCrossentropy
from .model import Model, serialize_attr, deserialize_attr
from .model import set_dropout_rate, change_attr_values, wrap_model_recursive
from .shims import Shim, PyTorchGradScaler, PyTorchShim, TensorFlowShim, keras_model_fns
from .shims import MXNetShim, maybe_handshake_model
from .optimizers import Adam, RAdam, SGD, Optimizer
from .schedules import cyclic_triangular, warmup_linear, constant, constant_then
from .schedules import decaying, slanted_triangular, compounding
from .types import Ragged, Padded, ArgsKwargs, Unserializable
from .util import fix_random_seed, is_cupy_array, set_active_gpu
from .util import prefer_gpu, require_gpu, require_cpu
from .util import DataValidationError, data_validation
from .util import to_categorical, get_width, get_array_module, to_numpy
from .util import torch2xp, xp2torch, tensorflow2xp, xp2tensorflow, mxnet2xp, xp2mxnet
from .util import get_torch_default_device
from .compat import has_cupy
from .backends import get_ops, set_current_ops, get_current_ops, use_ops
from .backends import Ops, CupyOps, MPSOps, NumpyOps, set_gpu_allocator
from .backends import use_pytorch_for_gpu_memory, use_tensorflow_for_gpu_memory

from .layers import Dropout, Embed, expand_window, HashEmbed, LayerNorm, Linear
from .layers import Maxout, Mish, MultiSoftmax, Relu, softmax_activation, Softmax, LSTM
from .layers import CauchySimilarity, ParametricAttention, Logistic
from .layers import resizable, sigmoid_activation, Sigmoid, SparseLinear
from .layers import ClippedLinear, ReluK, HardTanh, HardSigmoid
from .layers import Dish, HardSwish, HardSwishMobilenet, Swish, Gelu
from .layers import PyTorchWrapper, PyTorchRNNWrapper, PyTorchLSTM
from .layers import TensorFlowWrapper, keras_subclass, MXNetWrapper
from .layers import PyTorchWrapper_v2, Softmax_v2

from .layers import add, bidirectional, chain, clone, concatenate, noop
from .layers import residual, uniqued, siamese, list2ragged, ragged2list
from .layers import map_list
from .layers import with_array, with_array2d
from .layers import with_padded, with_list, with_ragged, with_flatten
from .layers import with_reshape, with_getitem, strings2arrays, list2array
from .layers import list2ragged, ragged2list, list2padded, padded2list, remap_ids
from .layers import array_getitem, with_cpu, with_debug, with_nvtx_range
from .layers import with_signpost_interval
from .layers import tuplify

from .layers import reduce_first, reduce_last, reduce_max, reduce_mean, reduce_sum


# fmt: off
__all__ = [
    # .config
    "Config", "registry", "ConfigValidationError",
    # .initializers
    "normal_init", "uniform_init", "glorot_uniform_init", "zero_init",
    "configure_normal_init",
    # .loss
    "CategoricalCrossentropy", "L2Distance", "CosineDistance",
    "SequenceCategoricalCrossentropy",
    # .model
    "Model", "serialize_attr", "deserialize_attr",
    "set_dropout_rate", "change_attr_values", "wrap_model_recursive",
    # .shims
    "Shim", "PyTorchGradScaler", "PyTorchShim", "TensorFlowShim", "keras_model_fns",
    "MXNetShim", "maybe_handshake_model",
    # .optimizers
    "Adam", "RAdam", "SGD", "Optimizer",
    # .schedules
    "cyclic_triangular", "warmup_linear", "constant", "constant_then",
    "decaying", "slanted_triangular", "compounding",
    # .types
    "Ragged", "Padded", "ArgsKwargs", "Unserializable",
    # .util
    "fix_random_seed", "is_cupy_array", "set_active_gpu",
    "prefer_gpu", "require_gpu", "require_cpu",
    "DataValidationError", "data_validation",
    "to_categorical", "get_width", "get_array_module", "to_numpy",
    "torch2xp", "xp2torch", "tensorflow2xp", "xp2tensorflow", "mxnet2xp", "xp2mxnet",
    "get_torch_default_device",
    # .compat
    "has_cupy",
    # .backends
    "get_ops", "set_current_ops", "get_current_ops", "use_ops",
    "Ops", "CupyOps", "MPSOps", "NumpyOps", "set_gpu_allocator",
    "use_pytorch_for_gpu_memory", "use_tensorflow_for_gpu_memory",
    # .layers
    "Dropout", "Embed", "expand_window", "HashEmbed", "LayerNorm", "Linear",
    "Maxout", "Mish", "MultiSoftmax", "Relu", "softmax_activation", "Softmax", "LSTM",
    "CauchySimilarity", "ParametricAttention", "Logistic",
    "resizable", "sigmoid_activation", "Sigmoid", "SparseLinear",
    "ClippedLinear", "ReluK", "HardTanh", "HardSigmoid",
    "Dish", "HardSwish", "HardSwishMobilenet", "Swish", "Gelu",
    "PyTorchWrapper", "PyTorchRNNWrapper", "PyTorchLSTM",
    "TensorFlowWrapper", "keras_subclass", "MXNetWrapper",
    "PyTorchWrapper_v2", "Softmax_v2",

    "add", "bidirectional", "chain", "clone", "concatenate", "noop",
    "residual", "uniqued", "siamese", "list2ragged", "ragged2list",
    "map_list",
    "with_array", "with_array2d",
    "with_padded", "with_list", "with_ragged", "with_flatten",
    "with_reshape", "with_getitem", "strings2arrays", "list2array",
    "list2ragged", "ragged2list", "list2padded", "padded2list", "remap_ids",
    "array_getitem", "with_cpu", "with_debug", "with_nvtx_range",
    "with_signpost_interval",
    "tuplify",

    "reduce_first", "reduce_last", "reduce_max", "reduce_mean", "reduce_sum",
]
# fmt: on
