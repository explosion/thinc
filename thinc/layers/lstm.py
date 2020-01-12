from typing import Optional, List, Tuple, Callable, cast

from ..model import Model
from ..backends import Ops
from ..config import registry
from ..util import get_width, has_torch
from ..types import RNNState, Array2d, Array3d, Padded
from .recurrent import recurrent
from .bidirectional import bidirectional
from .clone import clone
from .linear import Linear
from .noop import noop
from .with_padded import with_padded


InT = List[Array2d]


@registry.layers("LSTM.v0")
def LSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    bi: bool = False,
    depth: int = 1,
    dropout: float = 0.0
) -> Model[Padded, Padded]:
    if bi:
        if nO is not None:
            nO //= 2
        model = with_padded(
            clone(
                bidirectional(recurrent(LSTM_step(nO=nO, nI=nI, dropout=dropout))),
                depth
            )
        )
    else:
        model = with_padded(clone(recurrent(LSTM_step(nO=nO, nI=nI, dropout=dropout)), depth))
    return cast(Model[Padded, Padded], model)


@registry.layers("PyTorchLSTM.v0")
def PyTorchLSTM(
    nO: int,
    nI: int,
    *,
    bi: bool = False,
    depth: int = 1,
    dropout: float = 0.0
) -> Model[Padded, Padded]:
    import torch.nn
    from .with_padded import with_padded
    from .pytorchwrapper import PyTorchRNNWrapper

    if depth == 0:
        return noop()  # type: ignore

    return with_padded(
        PyTorchRNNWrapper(
            torch.nn.LSTM(nI, nO // 2, depth, bidirectional=bi, dropout=dropout)
        )
    )


def LSTM_step(
    nO: Optional[int] = None, nI: Optional[int] = None, *, dropout: float = 0.0
) -> Model[RNNState, RNNState]:
    """Create a step model for an LSTM."""
    if dropout != 0.0:
        msg = (
            "LSTM dropout not implemented yet. In the meantime, use the "
            "PyTorchWrapper and the torch.LSTM class."
        )
        raise NotImplementedError(msg)
    model: Model[RNNState, RNNState] = Model(
        "lstm_step", forward, init=init, layers=[Linear()], dims={"nO": nO, "nI": nI}
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def init(
    model: Model, X: Optional[RNNState] = None, Y: Optional[RNNState] = None
) -> None:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    model.layers[0].set_dim("nO", nO * 4)
    model.layers[0].set_dim("nI", nO + nI)
    model.layers[0].initialize()


def forward(
    model: Model[RNNState, RNNState], prevstate_inputs: RNNState, is_train: bool
) -> Tuple[RNNState, Callable]:
    (cell_tm1, hidden_tm1), inputs = prevstate_inputs
    weights = model.layers[0]
    nI = inputs.shape[1]
    X = model.ops.xp.hstack((inputs, hidden_tm1))

    acts, bp_acts = weights(X, is_train)
    (cells, hiddens), bp_gates = _gates_forward(model.ops, acts, cell_tm1)

    def backprop(d_state_d_hiddens: RNNState) -> RNNState:
        (d_cells, d_hiddens), d_hiddens = d_state_d_hiddens
        d_acts, d_cell_tm1 = bp_gates(d_cells, d_hiddens)
        dX = bp_acts(d_acts)
        return (d_cell_tm1, dX[:, nI:]), dX[:, :nI]

    return ((cells, hiddens), hiddens), backprop


def _gates_forward(ops: Ops, acts: Array3d, prev_cells: Array2d):
    nB = acts.shape[0]
    nO = acts.shape[1] // 4
    acts = acts.reshape((nB, nO, 4))
    new_cells = ops.alloc_f2d(*prev_cells.shape)
    new_hiddens = ops.alloc_f2d(*prev_cells.shape)

    ops.lstm(new_hiddens, new_cells, acts, prev_cells)
    size = new_cells.shape[0]

    def backprop_gates(d_cells: Array2d, d_hiddens: Array2d) -> Tuple[Array2d, Array2d]:
        d_cells = ops.xp.ascontiguousarray(d_cells[:size])
        d_hiddens = ops.xp.ascontiguousarray(d_hiddens[:size])
        d_acts = ops.alloc_f3d(*acts.shape)
        d_prevcells: Array2d = ops.alloc(prev_cells.shape)
        ops.backprop_lstm(
            d_cells, d_prevcells, d_acts, d_hiddens, acts, new_cells, prev_cells
        )
        d_reshaped: Array2d = d_acts.reshape((nB, nO * 4))
        return d_reshaped, d_prevcells

    return (new_cells, new_hiddens), backprop_gates
