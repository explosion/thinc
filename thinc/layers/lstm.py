from typing import Optional, List, Tuple, Callable, cast

from ..model import Model
from ..backends import Ops
from ..config import registry
from ..util import get_width
from ..types import Array, RNNState, Floats2d, FloatsNd
from .recurrent import recurrent
from .bidirectional import bidirectional
from .clone import clone
from .linear import Linear
from .noop import noop
from .with_list2padded import with_list2padded


InT = List[Floats2d]


@registry.layers("PyTorchBiLSTM.v0")
def PyTorchBiLSTM(nO, nI, depth, dropout=0.0):
    import torch.nn
    from .with_list2padded import with_list2padded
    from .pytorchwrapper import PyTorchWrapper

    if depth == 0:
        return noop()
    pytorch_lstm = torch.nn.LSTM(
        nI, nO // 2, depth, bidirectional=True, dropout=dropout
    )
    return with_list2padded(PyTorchWrapper(pytorch_lstm))


@registry.layers("BiLSTM.v0")
def BiLSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    depth: int = 1,
    dropout: float = 0.0
) -> Model[InT, InT]:
    return cast(
        Model[InT, InT],
        with_list2padded(
            clone(
                bidirectional(recurrent(LSTM_step(nO=nO, nI=nI, dropout=dropout))),
                depth,
            )
        ),
    )


@registry.layers("LSTM.v0")
def LSTM(
    nO: Optional[int] = None,
    nI: Optional[int] = None,
    *,
    depth: int = 1,
    dropout: float = 0.0
) -> Model[InT, InT]:
    return cast(
        Model[InT, InT],
        with_list2padded(
            clone(recurrent(LSTM_step(nO=nO, nI=nI, dropout=dropout)), depth)
        ),
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


def _gates_forward(ops: Ops, acts: Array, prev_cells: Floats2d):
    nB = acts.shape[0]
    nO = acts.shape[1] // 4
    acts = acts.reshape((nB, nO, 4))
    new_cells: FloatsNd = ops.allocate_nd(prev_cells.shape)
    new_hiddens: FloatsNd = ops.allocate_nd(prev_cells.shape)

    ops.lstm(new_hiddens, new_cells, acts, prev_cells)
    size = new_cells.shape[0]

    def backprop_gates(
        d_cells: Floats2d, d_hiddens: Floats2d
    ) -> Tuple[Floats2d, Floats2d]:
        d_cells = d_cells[:size]
        d_hiddens = d_hiddens[:size]
        d_acts: Floats2d = ops.allocate_nd(acts.shape)
        d_prevcells: Floats2d = ops.allocate_nd(prev_cells.shape)
        ops.backprop_lstm(
            d_cells, d_prevcells, d_acts, d_hiddens, acts, new_cells, prev_cells
        )
        d_acts = d_acts.reshape((nB, nO * 4))
        return d_acts, d_prevcells

    return (new_cells, new_hiddens), backprop_gates
