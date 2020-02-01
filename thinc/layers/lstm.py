from typing import Optional, Tuple, Callable, cast
from functools import partial

from ..model import Model
from ..config import registry
from ..util import get_width
from ..types import Floats1d, Floats2d, Floats3d, Padded, Ragged
from .bidirectional import bidirectional
from .clone import clone
from .noop import noop
from ..initializers import glorot_uniform_init, zero_init


@registry.layers("LSTM.v1")
def LSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    bi: bool = False,
    depth: int = 1,
    dropout: float = 0.0,
    init_W=glorot_uniform_init,
    init_b=zero_init
) -> Model[Ragged, Ragged]:
    if dropout != 0.0:
        msg = (
            "LSTM dropout not implemented yet. In the meantime, use the "
            "PyTorchWrapper and the torch.LSTM class."
        )
        raise NotImplementedError(msg)
    if depth == 0:
        msg = "LSTM depth must be at least 1. Maybe we should make this a noop?"
        raise ValueError(msg)

    if bi and nO is not None:
        nO //= 2
    model: Model[Ragged, Ragged] = Model(
        "lstm",
        forward,
        dims={"nO": nO, "nI": nI, "depth": depth, "dirs": 1+int(bi)},
        attrs={"registry_name": "LSTM.v1"},
        params={"LSTM": None, "HC0": None},
        init=partial(init, init_W, init_b),
    )
    return model


@registry.layers("PyTorchLSTM.v1")
def PyTorchLSTM(
    nO: int, nI: int, *, bi: bool = False, depth: int = 1, dropout: float = 0.0
) -> Model[Padded, Padded]:
    import torch.nn
    from .with_padded import with_padded
    from .pytorchwrapper import PyTorchRNNWrapper

    if depth == 0:
        return noop()  # type: ignore
    if bi:
        nO = nO // 2
    return with_padded(
        PyTorchRNNWrapper(
            torch.nn.LSTM(nI, nO, depth, bidirectional=bi, dropout=dropout)
        )
    )


def init(
    init_W: Callable,
    init_b: Callable,
    model: Model,
    X: Optional[Padded] = None,
    Y: Optional[Padded] = None,
) -> None:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    depth = model.get_dim("depth")
    dirs = model.get_dim("dirs")
    # It's easiest to use the initializer if we alloc the weights separately
    # and then stick them all together afterwards. The order matters here:
    # we need to keep the same format that CuDNN expects.
    params = []
    # Convenience
    init_W = partial(init_W, model.ops)
    init_b = partial(init_b, model.ops)
    for i in range(depth):
        for j in range(dirs):
            # Input-to-gates weights and biases.
            params.append(init_W(nO, nI if depth == 0 else nO))
            params.append(init_b(nO))
            params.append(init_W(nO, nI if depth == 0 else nO))
            params.append(init_b(nO))
            params.append(init_W(nO, nI if depth == 0 else nO))
            params.append(init_b(nO))
            params.append(init_W(nO, nI if depth == 0 else nO))
            params.append(init_b(nO))
            # Hidden-to-gates weights.
            params.append(init_W(nO, nO))
            params.append(init_b(nO))
            params.append(init_W(nO, nO))
            params.append(init_b(nO))
            params.append(init_W(nO, nO))
            params.append(init_b(nO))
            params.append(init_W(nO, nO))
            params.append(init_b(nO))
    model.set_param("LSTM", model.ops.xp.concatenate([p.ravel() for p in params])) 
    model.set_param("HC0", zero_init(model.ops, (2, depth, nO)))


def forward(
    model: Model[Ragged, Ragged], Xr: Ragged, is_train: bool
) -> Tuple[Ragged, Callable]:
    LSTM = cast(Floats1d, model.get_param("LSTM"))
    HC0 = cast(Floats3d, model.get_param("HC0"))
    H0 = HC0[0]
    C0 = HC0[1]
    if is_train:
        Y, fwd_state = model.ops.lstm_forward_training(
            LSTM, H0, C0, cast(Floats2d, Xr.data), Xr.lengths)
    else:
        Y = model.ops.lstm_forward_inference(LSTM, H0, C0, cast(Floats2d, Xr.data), Xr.lengths)
    Yr = Ragged(Y, Xr.lengths)

    def backprop(dYr: Ragged) -> Ragged:
        dX, (dLSTM, dH0, dC0) = model.ops.backprop_lstm(
            cast(Floats2d, dYr.data), fwd_state, LSTM
        )
        model.inc_grad("LSTM", dLSTM)
        model.inc_grad("HC0", HC0)
        return Ragged(dX, dYr.lengths)

    return Yr, backprop
