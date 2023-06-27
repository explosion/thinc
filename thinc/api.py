from .backends import (
    CupyOps,
    MPSOps,
    NumpyOps,
    Ops,
    get_current_ops,
    get_ops,
    set_current_ops,
    set_gpu_allocator,
    use_ops,
    use_pytorch_for_gpu_memory,
    use_tensorflow_for_gpu_memory,
)
from .compat import has_cupy
from .config import Config, ConfigValidationError, registry
from .initializers import (
    configure_normal_init,
    glorot_uniform_init,
    normal_init,
    uniform_init,
    zero_init,
)
from .layers import (
    LSTM,
    CauchySimilarity,
    ClippedLinear,
    Dish,
    Dropout,
    Embed,
    Gelu,
    HardSigmoid,
    HardSwish,
    HardSwishMobilenet,
    HardTanh,
    HashEmbed,
    LayerNorm,
    Linear,
    Logistic,
    Maxout,
    Mish,
    MultiSoftmax,
    MXNetWrapper,
    ParametricAttention,
    PyTorchLSTM,
    PyTorchRNNWrapper,
    PyTorchWrapper,
    PyTorchWrapper_v2,
    PyTorchWrapper_v3,
    Relu,
    ReluK,
    Sigmoid,
    Softmax,
    Softmax_v2,
    SparseLinear,
    SparseLinear_v2,
    Swish,
    TensorFlowWrapper,
    TorchScriptWrapper_v1,
    add,
    array_getitem,
    bidirectional,
    chain,
    clone,
    concatenate,
    expand_window,
    keras_subclass,
    list2array,
    list2padded,
    list2ragged,
    map_list,
    noop,
    padded2list,
    premap_ids,
    pytorch_to_torchscript_wrapper,
    ragged2list,
    reduce_first,
    reduce_last,
    reduce_max,
    reduce_mean,
    reduce_sum,
    remap_ids,
    remap_ids_v2,
    residual,
    resizable,
    siamese,
    sigmoid_activation,
    softmax_activation,
    strings2arrays,
    tuplify,
    uniqued,
    with_array,
    with_array2d,
    with_cpu,
    with_debug,
    with_flatten,
    with_flatten_v2,
    with_getitem,
    with_list,
    with_nvtx_range,
    with_padded,
    with_ragged,
    with_reshape,
    with_signpost_interval,
)
from .loss import (
    CategoricalCrossentropy,
    CosineDistance,
    L2Distance,
    SequenceCategoricalCrossentropy,
)
from .model import (
    Model,
    change_attr_values,
    deserialize_attr,
    serialize_attr,
    set_dropout_rate,
    wrap_model_recursive,
)
from .optimizers import SGD, Adam, Optimizer, RAdam
from .schedules import (
    compounding,
    constant,
    constant_then,
    cyclic_triangular,
    decaying,
    slanted_triangular,
    warmup_linear,
)
from .shims import (
    MXNetShim,
    PyTorchGradScaler,
    PyTorchShim,
    Shim,
    TensorFlowShim,
    TorchScriptShim,
    keras_model_fns,
    maybe_handshake_model,
)
from .types import ArgsKwargs, Padded, Ragged, Unserializable
from .util import (
    DataValidationError,
    data_validation,
    fix_random_seed,
    get_array_module,
    get_torch_default_device,
    get_width,
    is_cupy_array,
    mxnet2xp,
    prefer_gpu,
    require_cpu,
    require_gpu,
    set_active_gpu,
    tensorflow2xp,
    to_categorical,
    to_numpy,
    torch2xp,
    xp2mxnet,
    xp2tensorflow,
    xp2torch,
)

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
    "MXNetShim", "TorchScriptShim", "maybe_handshake_model",
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
    "PyTorchWrapper_v2", "Softmax_v2", "PyTorchWrapper_v3",
    "SparseLinear_v2", "TorchScriptWrapper_v1",

    "add", "bidirectional", "chain", "clone", "concatenate", "noop",
    "residual", "uniqued", "siamese", "list2ragged", "ragged2list",
    "map_list",
    "with_array", "with_array2d",
    "with_padded", "with_list", "with_ragged", "with_flatten",
    "with_reshape", "with_getitem", "strings2arrays", "list2array",
    "list2ragged", "ragged2list", "list2padded", "padded2list", 
    "remap_ids", "remap_ids_v2", "premap_ids",
    "array_getitem", "with_cpu", "with_debug", "with_nvtx_range",
    "with_signpost_interval",
    "tuplify", "with_flatten_v2",
    "pytorch_to_torchscript_wrapper",

    "reduce_first", "reduce_last", "reduce_max", "reduce_mean", "reduce_sum",
]
# fmt: on
