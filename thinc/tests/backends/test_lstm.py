from thinc.backends.jax_ops import lstm_weights_forward, backprop_lstm_weights
from thinc.backends.jax_ops import lstm_gates_forward, backprop_lstm_gates
import numpy.testing
import pytest

from hypothesis import given, settings
from ..strategies import ndarrays_of_shape

try:
    import jax

    has_jax = True
except ImportError:
    has_jax = False

MAX_EXAMPLES = 20

nL = 6
nB = 3
nO = 4
nI = 2
t = 3


def assert_arrays_equal(arrays1, arrays2):
    assert len(arrays1) == len(arrays2)
    shapes1 = [tuple(a.shape) for a in arrays1]
    shapes2 = [tuple(a.shape) for a in arrays2]
    assert shapes1 == shapes2
    for arr1, arr2 in zip(arrays1, arrays2):
        assert arr1.shape == arr2.shape
        numpy.testing.assert_allclose(arr1, arr2, rtol=0.001, atol=0.001)


# See thinc/backends/jax_ops for notation


@pytest.mark.skipif(not has_jax, reason="needs Jax")
@pytest.mark.filterwarnings("ignore")
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    Xt3=ndarrays_of_shape((nB, nI), dtype="f"),
    Yt2=ndarrays_of_shape((nB, nO), dtype="f"),
    dAt3=ndarrays_of_shape((nB, nO * 4), dtype="f"),
    W=ndarrays_of_shape((nO * 4, nO + nI), dtype="f"),
    b=ndarrays_of_shape((nO * 4,), dtype="f"),
)
def test_lstm_weights_gradients(Xt3, Yt2, W, b, dAt3):
    At3, jax_backprop = jax.vjp(lstm_weights_forward, Xt3, Yt2, W, b)
    jax_grads = jax_backprop(dAt3)
    St3 = jax.numpy.hstack((Xt3, Yt2))
    our_grads = backprop_lstm_weights(dAt3, (St3, W, b))
    assert_arrays_equal(our_grads, jax_grads)


@pytest.mark.skipif(not has_jax, reason="needs Jax")
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    At3=ndarrays_of_shape((nB, nO * 4), dtype="f"),
    Ct2=ndarrays_of_shape((nB, nO), dtype="f"),
    dYt3=ndarrays_of_shape((nB, nO), dtype="f"),
    dCt3=ndarrays_of_shape((nB, nO), dtype="f"),
)
def test_lstm_gates_gradients(At3, Ct2, dYt3, dCt3):
    # At3 = (At3 * 0) + 1
    # Ct2 = (Ct2 * 0) + 1
    # dYt3 = (dYt3 * 0) + 1
    # dCt3 = (dCt3 * 0) + 1
    (Yt3, Ct3, Gt3), get_jax_grads = jax.vjp(lstm_gates_forward, At3, Ct2)
    jax_grads = get_jax_grads((dYt3, dCt3, Gt3 * 0))
    Yt3, Ct3, Gt3 = lstm_gates_forward(At3, Ct2)
    our_grads = backprop_lstm_gates(dYt3, dCt3, Gt3, Ct3, Ct2)
    assert_arrays_equal(our_grads, jax_grads)
