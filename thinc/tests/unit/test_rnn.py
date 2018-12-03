# coding: utf8
from __future__ import unicode_literals

import numpy
import timeit

from ...neural.util import minibatch
from ...neural.ops import NumpyOps
from ...neural._classes.rnn import LSTM


def test_square_sequences():
    ops = NumpyOps()
    seqs = [numpy.zeros((5, 4)), numpy.zeros((8, 4)), numpy.zeros((2, 4))]
    arr, size_at_t, unpad = ops.square_sequences(seqs)
    assert arr.shape == (8, 3, 4)
    assert size_at_t[0] == 3
    assert size_at_t[1] == 3
    assert size_at_t[2] == 2
    assert size_at_t[3] == 2
    assert size_at_t[4] == 2
    assert size_at_t[5] == 1
    assert size_at_t[6] == 1
    assert size_at_t[7] == 1
    unpadded = unpad(arr)
    assert unpadded[0].shape == (5, 4)
    assert unpadded[1].shape == (8, 4)
    assert unpadded[2].shape == (2, 4)


def test_LSTM_init():
    model = LSTM(1, 2)
    model = LSTM(2, 2)
    model = LSTM(100, 200)
    model = LSTM(9, 6)  # noqa: F841


def test_LSTM_fwd_bwd_shapes():
    nO = 1
    nI = 2
    model = LSTM(nO, nI)

    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype="f")
    ys, backprop_ys = model.begin_update([X])
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape


# def test_RNN_fwd_correctness():
#    nO = 1
#    nI = 2
#    alloc, params = numpy_params()
#    model = _RNN(alloc, nO, nI, begin_stepwise_relu, nG=1)
#    (Wx, dWx), (Wh, dWh), (b, db), (pad, d_pad) = params
#
#    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
#    Wx[:] = 1.
#    b[:] = 0.25
#    Wh[:] = 1.
#    pad[:] = 0.05
#
#    # Step 1
#    # (0.1, 0.1) @ [[1., 1.]] + (1.,) @ [[0.05]]
#    # = (0.1 * 1) * 2 + 0.05
#    # = (0.25,) + 0.25 bias
#    Y = model([X])
#    Y = Y[0][0]
#    assert list(Y[0]) == [0.5]
#
#
# @pytest.mark.skip
# def test_RNN_learns():
#    nO = 2
#    nI = 2
#    alloc, params = numpy_params()
#    model = _RNN(alloc, nO, nI)
#    X = numpy.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype='f')
#    Y = numpy.asarray([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype='f')
#    Yhs, bp_Yhs = model([X])
#    dXs = bp_Yhs([Yhs[0] - Y])
#    loss1 = ((Yhs[0]-Y)**2).sum()
#    for param, grad in params:
#        param -= 0.001 * grad
#        grad.fill(0)
#    Yhs, bp_Yhs = model([X])
#    dXs = bp_Yhs([Yhs[0] - Y])
#    loss2 = ((Yhs[0]-Y)**2).sum()
#    assert loss1 > loss2, (loss1, loss2)
#
#
#
def test_LSTM_learns():
    nO = 2
    nI = 2
    model = LSTM(nO, nI)

    def sgd(weights, gradient, key=None):
        weights -= 0.001 * gradient
        gradient.fill(0.0)

    X = numpy.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype="f")
    Y = numpy.asarray([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype="f")
    Yhs, bp_Yhs = model.begin_update([X])
    loss1 = ((Yhs[0] - Y) ** 2).sum()
    Yhs, bp_Yhs = model.begin_update([X])
    dXs = bp_Yhs([Yhs[0] - Y], sgd=sgd)
    Yhs, bp_Yhs = model.begin_update([X])
    dXs = bp_Yhs([Yhs[0] - Y], sgd=sgd)  # noqa: F841
    loss2 = ((Yhs[0] - Y) ** 2).sum()
    assert loss1 > loss2, (loss1, loss2)


# def test_LSTM_fwd():
#    nO = 2
#    nI = 2
#    alloc, params = numpy_params()
#    model = _BiLSTM(alloc, nO, nI)
#
#    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
#    ys, backprop_ys =  model([X])
#    dXs = backprop_ys(ys)
#    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape
#
#
def test_benchmark_RNN_fwd():
    nO = 128
    nI = 128
    n_batch = 1000
    batch_size = 30
    seq_len = 30
    lengths = numpy.random.normal(scale=10, loc=30, size=n_batch * batch_size)
    lengths = numpy.maximum(lengths, 1)
    batches = []
    uniform_lengths = False
    for batch_lengths in minibatch(lengths, batch_size):
        batch_lengths = list(batch_lengths)
        if uniform_lengths:
            seq_len = max(batch_lengths)
            batch = [
                numpy.asarray(
                    numpy.random.uniform(0.0, 1.0, (int(seq_len), nI)), dtype="f"
                )
                for _ in batch_lengths
            ]
        else:
            batch = [
                numpy.asarray(
                    numpy.random.uniform(0.0, 1.0, (int(seq_len), nI)), dtype="f"
                )
                for seq_len in batch_lengths
            ]
        batches.append(batch)
    model = LSTM(nO, nI)
    start = timeit.default_timer()
    for Xs in batches:
        ys, bp_ys = model.begin_update(list(Xs))
        # _ = bp_ys(ys)
    end = timeit.default_timer()
    n_samples = n_batch * batch_size
    print(
        "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---"
        % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples)
    )
