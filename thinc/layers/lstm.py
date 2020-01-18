from typing import Optional, Tuple, Callable
from functools import partial

from ..model import Model
from ..backends import Ops
from ..config import registry
from ..util import get_width
from ..types import RNNState, Array2d, Array3d, Padded
from .bidirectional import bidirectional
from .clone import clone
from .linear import Linear
from .noop import noop
from ..initializers import xavier_uniform_init, zero_init


@registry.layers("LSTM.v0")
def LSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    bi: bool = False,
    depth: int = 1,
    dropout: float = 0.0,
    init_W=xavier_uniform_init,
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
        params={"W": None, "b": None, "c": None, "h": None},
        init=partial(init, init_W, init_b)
    )

    if bi:
        model = bidirectional(model)
    return clone(model, depth)


@registry.layers("PyTorchLSTM.v0")
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
        init_W: Callable, init_b: Callable, model: Model,
        X: Optional[Padded] = None, Y: Optional[Padded] = None
) -> None:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    model.set_param("W", init_W(model.ops, (nO*4, nO+nI)))
    model.set_param("b", init_b(model.ops, (nO*4,)))
    model.set_param("h", zero_init(model.ops, (nO,)))
    model.set_param("c", zero_init(model.ops, (nO,)))


def forward(
    model: Model[Array3d, Array3d], Xp: Padded, is_train: bool
) -> Tuple[Padded, Callable]:
    X = Xp.data
    W = model.get_param("W")
    b = model.get_param("b")
    h = model.get_param("h")
    c = model.get_param("c")
    # Initialize hiddens and cells
    hiddens = model.ops.alloc_f2d(X.shape[1], h.shape[0])
    cells = model.ops.alloc_f2d(X.shape[1], c.shape[0])
    hiddens += h
    cells += c
    Y, cells, gates = model.ops.recurrent_lstm(W, b, hiddens, cells, X)
    Yp = Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices)

    def backprop(dYp: Padded) -> Padded:
        raise NotImplementedError

    return Yp, backprop


"""
def backprop_gates(d_cells: Array2d, d_hiddens: Array2d) -> Tuple[Array3d, Array2d]:
    d_cells = ops.as_contig(d_cells[:size])  # Wtf?
    d_hiddens = ops.as_contig(d_hiddens[:size])
    d_acts, d_prevcells = ops.backprop_lstm(
        d_cells, d_hiddens, gates, cells, prevcells
    )
    d_acts = d_acts.reshape((nB, nO * 4))
    return d_acts, d_prevcells
"""
