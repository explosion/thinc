from typing import Tuple
from ..model import Model
from ..types import Padded


def recurrent(step_model: Model) -> Model:
    model = Model(
        step_model.name.replace("_step", ""),
        forward,
        init=init,
        params={"initial_cells": None, "initial_hiddens": None},
        dims={"nO": step_model.get_dim("nO") if step_model.has_dim("nO") else None},
        layers=[step_model],
    )
    if model.has_dim("nO"):
        model.initialize()
    return model


def init(model, X=None, Y=None):
    Xt = X[0] if X is not None else None
    Yt = Y[0] if Y is not None else None
    if Xt is not None or Yt is not None:
        model.layers[0].initialize(X=Xt, Y=Yt)
    nO = model.get_dim("nO")
    model.set_param("initial_cells", model.ops.allocate((nO,)))
    model.set_param("initial_hiddens", model.ops.allocate((nO,)))


def forward(model, Xp, is_train):
    # Expect padded batches, sorted by decreasing length. The size_at_t array
    # records the number of batch items that are still active at timestep t.
    X = Xp.data
    size_at_t = Xp.size_at_t
    step_model = model.layers[0]
    nI = step_model.get_dim("nI")
    nO = step_model.get_dim("nO")
    Y = model.ops.allocate((X.shape[0], X.shape[1], nO))
    backprops = [None] * X.shape[0]
    (cell, hidden) = _get_initial_state(model, X.shape[1], nO)
    for t in range(X.shape[0]):
        # At each timestep t, we finish some of the sequences. The sequences
        # are arranged longest to shortest, so we can drop the finished ones
        # off the end.
        n = size_at_t[t]
        inputs = ((cell[:n], hidden[:n]), X[t, :n])
        ((cell, hidden), Y[t, :n]), backprops[t] = step_model(inputs, is_train)

    def backprop(dYp):
        dY = dYp.data
        size_at_t = dYp.size_at_t
        d_state = (
            step_model.ops.allocate((dY.shape[1], nO)),
            step_model.ops.allocate((dY.shape[1], nO)),
        )
        dX = step_model.ops.allocate((dY.shape[0], dY.shape[1], nI))
        for t in range(dX.shape[0] - 1, -1, -1):
            n = size_at_t[t]
            d_state, dX[t, :n] = backprops[t]((d_state, dY[t]))
        model.inc_grad("initial_cells", d_state[0].sum(axis=0))
        model.inc_grad("initial_hiddens", d_state[1].sum(axis=0))
        return Padded(dX, size_at_t)

    return Padded(Y, size_at_t), backprop


def _get_initial_state(model, n, nO):
    initial_cells = model.ops.allocate((n, nO))
    initial_hiddens = model.ops.allocate((n, nO))
    initial_cells += model.get_param("initial_cells")
    initial_hiddens += model.get_param("initial_hiddens")
    return (initial_cells, initial_hiddens)
