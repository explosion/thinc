from typing import Tuple
from ..model import Model
from ..types import Array


def recurrent(step_model: Model) -> Model:
    return Model(step_model.name, forward, layers=[step_model])


def forward(model: Model, X_size_at_t: Tuple[Array, Array], is_train: bool):
    # Expect padded batches, sorted by decreasing length. The size_at_t array
    # records the number of batch items that are still active at timestep t.
    X, size_at_t = X_size_at_t
    step_model = model.layers[0]
    nO = step_model.get_dim("nO")
    Y = model.ops.allocate((X.shape[0], X.shape[1], nO))
    backprops = [None] * X.shape[0]
    state = _get_initial_state(step_model, X.shape[1], nO)
    for t in range(X.shape[0]):
        state = list(state)
        size = size_at_t[t]
        Xt = X[t, :size]
        state[0] = state[0][:size]
        state[1] = state[1][:size]
        inputs = (state, Xt)
        (state, Y[t, :size]), backprops[t] = step_model(inputs, is_train)

    def backprop(dY_size_at_t):
        dY, size_at_t = dY_size_at_t
        d_state = [
            step_model.ops.allocate((dY.shape[1], step_model.nO)),
            step_model.ops.allocate((dY.shape[1], step_model.nO)),
        ]
        dX = step_model.ops.allocate((dY.shape[0], dY.shape[1], step_model.weights.nI))
        for t in range(dX.shape[0] - 1, -1, -1):
            d_state_t, dXt = backprops[t]((d_state, dY[t]))
            d_state[0][: d_state_t[0].shape[0]] = d_state_t[0]
            d_state[1][: d_state_t[1].shape[0]] = d_state_t[1]
            dX[t, : dXt.shape[0]] = dXt
        d_cell, d_hidden = d_state
        step_model.inc_grad("initial_cells", d_cell.sum(axis=0))
        step_model.inc_grad("initial_hiddens", d_hidden.sum(axis=0))
        return (dX, size_at_t)

    return (Y, size_at_t), backprop


def _get_initial_state(step_model, n, nO):
    initial_cells = step_model.ops.allocate((n, nO))
    initial_hiddens = step_model.ops.allocate((n, nO))
    initial_cells += step_model.get_param("initial_cells")
    initial_hiddens += step_model.get_param("initial_hiddens")
    return (initial_cells, initial_hiddens)
