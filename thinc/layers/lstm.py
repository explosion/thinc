from ..model import Model
from ..util import get_width
from .recurrent import recurrent
from .bidirectional import bidirectional
from .clone import clone
from .affine import Affine


def BiLSTM(nO=None, nI=None, *, depth=1, dropout=0.0):
    return clone(bidirectional(recurrent(LSTM_step(nO=nO, nI=nI))), depth)


def LSTM(nO=None, nI=None, *, depth=1, dropout=0.0):
    return clone(recurrent(LSTM_step(nO=nO, nI=nI)), depth)


def LSTM_step(nO=None, nI=None):
    """Create a step model for an LSTM."""
    model = Model(
        "lstm_step", forward, init=init, layers=[Affine()], dims={"nO": nO, "nI": nI}
    )
    if nO is not None and nI is not None:
        model.initialize()
    return model


def init(model, X=None, Y=None):
    if X is not None:
        model.set_dim("nI", get_width(X))
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    model.layers[0].set_dim("nO", nO * 4)
    model.layers[0].set_dim("nI", nO + nI)
    model.layers[0].initialize()


def forward(model, prevstate_inputs, is_train):
    (cell_tm1, hidden_tm1), inputs = prevstate_inputs
    weights = model.layers[0]
    nI = model.get_dim("nI")
    nO = model.get_dim("nO")
    nB = inputs.shape[0]
    X = model.ops.xp.hstack((inputs, hidden_tm1))

    Y, bp_acts = weights(X, is_train)
    acts = Y.reshape((nB, 4, nO)).transpose((1, 0, 2))
    (cells, hiddens), bp_gates = _gates_forward(model.ops, acts, cell_tm1)

    def backprop(d_state_d_hiddens):
        (d_cells, d_hiddens), d_hiddens = d_state_d_hiddens
        d_acts, d_cell_tm1 = bp_gates(d_cells, d_hiddens)
        dY = d_acts.transpose((1, 0, 2)).reshape((nB, 4 * nO))
        dX = bp_acts(dY)
        return (d_cell_tm1, dX[:, nI:]), dX[:, :nI]

    return ((cells, hiddens), hiddens), backprop


def _gates_forward(ops, acts, prev_cells):
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
        return d_acts, d_prevcells

    return (new_cells, new_hiddens), backprop_gates
