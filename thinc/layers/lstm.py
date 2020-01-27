from typing import Optional, Tuple, Callable, cast
from functools import partial

from ..model import Model
from ..config import registry
from ..util import get_width
from ..types import Floats1d, Floats2d, Floats3d, Padded
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
) -> Model[Padded, Padded]:
    if dropout != 0.0:
        msg = (
            "LSTM dropout not implemented yet. In the meantime, use the "
            "PyTorchWrapper and the torch.LSTM class."
        )
        raise NotImplementedError(msg)

    if bi and nO is not None:
        nO //= 2
    model: Model[Padded, Padded] = Model(
        "lstm",
        forward,
        dims={"nO": nO, "nI": nI},
        attrs={"registry_name": "LSTM.v1"},
        params={"W": None, "b": None, "c": None, "h": None},
        init=partial(init, init_W, init_b),
    )

    if bi:
        model = bidirectional(model)
    return clone(model, depth)


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
    model.set_param("W", init_W(model.ops, (nO * 4, nO + nI)))
    model.set_param("b", init_b(model.ops, (nO * 4,)))
    model.set_param("h", zero_init(model.ops, (nO,)))
    model.set_param("c", zero_init(model.ops, (nO,)))


def forward(
    model: Model[Floats3d, Floats3d], Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    X = Xp.data
    W = cast(Floats2d, model.get_param("W"))
    b = cast(Floats1d, model.get_param("b"))
    h = cast(Floats1d, model.get_param("h"))
    c = cast(Floats1d, model.get_param("c"))
    Y, fwd_state = model.ops.recurrent_lstm(W, b, h, c, X, is_train)
    Yp = Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices)

    def backprop(dYp: Padded) -> Padded:
        dX, (dW, db, d_h, d_c) = model.ops.backprop_recurrent_lstm(
            dYp.data, fwd_state, (W, b)
        )
        model.inc_grad("W", dW)
        model.inc_grad("b", db)
        model.inc_grad("h", d_h)
        model.inc_grad("c", d_c)
        return Padded(X, dYp.size_at_t, dYp.lengths, dYp.indices)

    return Yp, backprop
