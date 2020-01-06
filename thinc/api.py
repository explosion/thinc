# flake8: noqa
from .loss import categorical_crossentropy, L1_distance, cosine_distance
from .config import Config
from .initializers import normal_init, uniform_init, xavier_uniform_init, zero_init
from .loss import categorical_crossentropy, L1_distance, cosine_distance
from .model import create_init, Model
from .optimizers import Adam, RAdam, SGD, Optimizer, ADAM_DEFAULTS, SGD_DEFAULTS
from .schedules import (
    cyclic_triangular,
    warmup_linear,
    constant,
    constant_then,
    decaying,
    warmup_linear,
    slanted_triangular,
    compounding,
)
from ._registry import registry
from .types import Array, Floats1d, Floats2d, Floats3d, Floats4d, FloatsNd
from .types import Ints1d, Ints2d, Ints3d, Ints4d, IntsNd
from .types import RNNState, Ragged, Padded, Xp, Shape, DTypes
from .util import (
    fix_random_seed,
    create_thread_local,
    is_cupy_array,
    get_ops,
    set_active_gpu,
    prefer_gpu,
    copy_array,
    get_shuffled_batches,
    minibatch,
    evaluate_model_on_arrays,
    to_categorical,
    get_width,
    xp2torch,
    torch2xp,
)
from .backends import (
    set_current_ops,
    get_current_ops,
    use_device,
    Ops,
    CupyOps,
    NumpyOps,
)
from .layers import (
    Affine,
    Dropout,
    Embed,
    ExtractWindow,
    HashEmbed,
    LayerNorm,
    Maxout,
    Mish,
    MultiSoftmax,
    ReLu,
    Residual,
    Softmax,
    BiLSTM,
    LSTM,
)
from .layers import (
    add,
    bidirectional,
    chain,
    clone,
    concatenate,
    foreach,
    noop,
    recurrent,
    uniqued,
    siamese,
    MaxPool,
    MeanPool,
    SumPool,
    list2ragged,
    ragged2list,
    with_list2array,
    with_list2padded,
    with_reshape,
)

__all__ = list(locals().keys())
