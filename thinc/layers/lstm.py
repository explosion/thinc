from typing import Optional

from ..model import Model
from ..types import Array
from ..util import get_width
from .recurrent import recurrent
from .bidirectional import bidirectional
from .clone import clone
from .affine import Affine
from .with_list2padded import with_list2padded


# TODO: finish types


def BiLSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    depth: int = 1,
    dropout: float = 0.0
):
    return with_list2padded(
        clone(bidirectional(recurrent(LSTM_step(nO=nO, nI=nI, dropout=dropout))), depth)
    )


def LSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    depth: int = 1,
    dropout: float = 0.0
):
    return with_list2padded(
        clone(recurrent(LSTM_step(nO=nO, nI=nI, dropout=dropout)), depth)
    )


def LSTM_step(
    nO: Optional[int] = None, nI: Optional[int] = None, *, dropout: float = 0.0
):
    """Create a step model for an LSTM."""
    if dropout != 0.0:
        msg = (
            "LSTM dropout not implemented yet. In the meantime, use the "
            "PyTorchWrapper and the torch.LSTM class."
        )
        raise NotImplementedError(msg)
    model = Model(
        "lstm_step", forward, init=init, layers=[Affine()], dims={"nO": nO, "nI": nI}
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def init(model: Model, X: Optional[Array] = None, Y: Optional[Array] = None) -> None:
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    model.layers[0].set_dim("nO", nO * 4)
    model.layers[0].set_dim("nI", nO + nI)
    model.layers[0].initialize()


def forward(model: Model, prevstate_inputs, is_train: bool):
    (cell_tm1, hidden_tm1), inputs = prevstate_inputs
    weights = model.layers[0]
    nI = inputs.shape[1]
    X = model.ops.xp.hstack((inputs, hidden_tm1))

    acts, bp_acts = weights(X, is_train)
    (cells, hiddens), bp_gates = _gates_forward(model.ops, acts, cell_tm1)

    def backprop(d_state_d_hiddens):
        (d_cells, d_hiddens), d_hiddens = d_state_d_hiddens
        d_acts, d_cell_tm1 = bp_gates(d_cells, d_hiddens)
        dX = bp_acts(d_acts)
        return (d_cell_tm1, dX[:, nI:]), dX[:, :nI]

    return ((cells, hiddens), hiddens), backprop


def _gates_forward(ops, acts, prev_cells):
    nB = acts.shape[0]
    nO = acts.shape[1] // 4
    acts = acts.reshape((nB, nO, 4))
    new_cells = ops.allocate(prev_cells.shape)
    new_hiddens = ops.allocate(prev_cells.shape)

    ops.lstm(new_hiddens, new_cells, acts, prev_cells)
    size = new_cells.shape[0]

    def backprop_gates(d_cells, d_hiddens):
        d_cells = d_cells[:size]
        d_hiddens = d_hiddens[:size]
        d_acts = ops.allocate(acts.shape)
        d_prevcells = ops.allocate(prev_cells.shape)
        ops.backprop_lstm(
            d_cells, d_prevcells, d_acts, d_hiddens, acts, new_cells, prev_cells
        )
        d_acts = d_acts.reshape((nB, nO * 4))
        return d_acts, d_prevcells

    return (new_cells, new_hiddens), backprop_gates
