from ..model import Model
from ..recurrent import recurrent
from ..bidirectional import bidirectional
from ..clone import clone


def BiLSTM(nO=None, nI=None, depth=1, dropout=0.0):
    return clone(bidirectional(recurrent(LSTM_step(nO=nO, nI=nI))), depth)


def LSTM(nO=None, nI=None, depth=1, dropout=0.0):
    return clone(recurrent(LSTM_step(nO=nO, nI=nI)), depth)


def LSTM_step(nO=None, nI=None):
    """Create a step model for an LSTM."""
    return Model(
        "lstm_step",
        forward,
        init=init,
        dims={"nO": nO, "nI": nI},
        params={"W": None, "b": None, "initial_hiddens": None, "initial_cells": None},
    )


def init(model, X=None, Y=None):
    pass


def forward(model, prevstate_inputs, is_train):
    (cell_tm1, hidden_tm1), inputs = prevstate_inputs

    acts, bp_acts = _weights_forward(model, (inputs, hidden_tm1), is_train)
    (cells, hiddens), bp_gates = _gates_forward(model, (acts, cell_tm1), is_train)

    def backprop(d_state_d_hiddens):
        (d_cells, d_hiddens), d_hiddens = d_state_d_hiddens
        d_acts, d_cell_tm1 = bp_gates((d_cells, d_hiddens))
        d_inputs, d_hidden_tm1 = bp_acts(d_acts)
        return (d_cell_tm1, d_hidden_tm1), d_inputs

    return ((cells, hiddens), hiddens), backprop


def _weights_forward(model, inputs_hidden, is_train):
    inputs, hidden = inputs_hidden
    nO = model.get_dim("nO")
    nI = model.get_dim("nI")
    W = model.get_param("W")
    b = model.get_param("b")
    forget_bias = model.get_param("forget_bias")
    X = model.ops.xp.hstack([inputs, hidden])
    acts = model.ops.gemm(X, W, trans2=True)
    acts += b
    acts = acts.reshape((acts.shape[0], 4, nO))
    acts = model.ops.xp.ascontiguousarray(acts.transpose((1, 0, 2)))

    acts[0] += forget_bias

    def backprop_weights(d_acts):
        model.inc_grad("forget_bias", d_acts[0].sum(axis=0))
        dX = model.ops.gemm(d_acts, W)
        model.inc_grad("W", model.ops.gemm(d_acts, X, trans1=True))
        model.inc_grad("b", d_acts.sum(axis=0))
        d_input = dX[:, :nI]
        d_hidden = dX[:, nI:]
        return d_input, d_hidden

    return acts, backprop_weights


def _gates_forward(model, acts_prev_cells, is_train):
    acts, prev_cells = acts_prev_cells
    new_cells = model.ops.allocate(prev_cells.shape)
    new_hiddens = model.ops.allocate(prev_cells.shape)
    model.ops.lstm(new_hiddens, new_cells, acts, prev_cells)
    state = (new_cells, new_hiddens)
    size = new_cells.shape[0]

    def backprop_gates(d_state):
        d_cells, d_hiddens = d_state
        d_cells = d_cells[:size]
        d_hiddens = d_hiddens[:size]
        d_acts = [model.ops.allocate(act.shape) for act in acts]
        d_prev_cells = model.ops.allocate(prev_cells.shape)
        model.ops.backprop_lstm(
            d_cells, d_prev_cells, d_acts, d_hiddens, acts, new_cells, prev_cells
        )
        return d_acts, d_prev_cells

    return state, backprop_gates
