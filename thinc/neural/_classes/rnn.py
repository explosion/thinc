# coding: utf8
from __future__ import unicode_literals

from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient
from ...api import wrap, layerize
from .._lsuv import svd_orthonormal
from ..util import copy_array


def BiLSTM(nO, nI):
    """Create a bidirectional LSTM layer. Args: number out, number in"""
    return Bidirectional(LSTM(nO // 2, nI), LSTM(nO // 2, nI))


def LSTM(nO, nI):
    """Create an LSTM layer. Args: number out, number in"""
    weights = LSTM_weights(nO, nI)
    gates = LSTM_gates(weights.ops)
    return Recurrent(RNN_step(weights, gates))


def Bidirectional(l2r, r2l):
    """Stitch two RNN models into a bidirectional layer."""
    nO = l2r.nO

    def birnn_fwd(Xs, drop=0.0):
        l2r_Zs, bp_l2r_Zs = l2r.begin_update(Xs, drop=drop)
        r2l_Zs, bp_r2l_Zs = r2l.begin_update(
            [l2r.ops.xp.ascontiguousarray(X[::-1]) for X in Xs]
        )

        def birnn_bwd(dZs, sgd=None):
            d_l2r_Zs = []
            d_r2l_Zs = []
            for dZ in dZs:
                l2r_fwd = dZ[:, :nO]
                r2l_fwd = dZ[:, nO:]
                d_l2r_Zs.append(l2r.ops.xp.ascontiguousarray(l2r_fwd))
                d_r2l_Zs.append(l2r.ops.xp.ascontiguousarray(r2l_fwd[::-1]))
            dXs_l2r = bp_l2r_Zs(d_l2r_Zs, sgd=sgd)
            dXs_r2l = bp_r2l_Zs(d_r2l_Zs, sgd=sgd)
            dXs = [dXf + dXb[::-1] for dXf, dXb in zip(dXs_l2r, dXs_r2l)]
            return dXs

        Zs = [l2r.ops.xp.hstack((Zf, Zb[::-1])) for Zf, Zb in zip(l2r_Zs, r2l_Zs)]
        return Zs, birnn_bwd

    return wrap(birnn_fwd, l2r, r2l)


def Recurrent(step_model):
    """Apply a stepwise model over a sequence, maintaining state. For RNNs"""
    ops = step_model.ops

    def recurrent_fwd(seqs, drop=0.0):
        lengths = [len(X) for X in seqs]
        X, size_at_t, unpad = ops.square_sequences(seqs)
        Y = ops.allocate((X.shape[0], X.shape[1], step_model.nO))
        cell_drop = ops.get_dropout_mask((len(seqs), step_model.nO), 0.0)
        hidden_drop = ops.get_dropout_mask((len(seqs), step_model.nO), 0.0)
        out_drop = ops.get_dropout_mask((len(seqs), step_model.nO), 0.0)
        backprops = [None] * max(lengths)
        state = step_model.weights.get_initial_state(len(seqs))
        for t in range(max(lengths)):
            state = list(state)
            size = size_at_t[t]
            Xt = X[t, :size]
            state[0] = state[0][:size]
            state[1] = state[1][:size]
            if cell_drop is not None:
                state[0] *= cell_drop
            if hidden_drop is not None:
                state[1] *= hidden_drop
            inputs = (state, Xt)
            (state, Y[t, :size]), backprops[t] = step_model.begin_update(inputs)
            if out_drop is not None:
                Y[t, :size] *= out_drop
        outputs = unpad(Y)

        def recurrent_bwd(d_outputs, sgd=None):
            dY, size_at_t, unpad = step_model.ops.square_sequences(d_outputs)
            d_state = [
                step_model.ops.allocate((dY.shape[1], step_model.nO)),
                step_model.ops.allocate((dY.shape[1], step_model.nO)),
            ]
            updates = {}

            def gather_updates(weights, gradient, key=None):
                updates[key] = (weights, gradient)

            dX = step_model.ops.allocate(
                (dY.shape[0], dY.shape[1], step_model.weights.nI)
            )
            for t in range(max(lengths) - 1, -1, -1):
                if out_drop is not None:
                    dY[t] *= out_drop
                d_state_t, dXt = backprops[t]((d_state, dY[t]), sgd=gather_updates)
                d_state[0][: d_state_t[0].shape[0]] = d_state_t[0]
                d_state[1][: d_state_t[1].shape[0]] = d_state_t[1]
                dX[t, : dXt.shape[0]] = dXt
                if cell_drop is not None:
                    d_state[0] *= cell_drop
                if hidden_drop is not None:
                    d_state[1] *= hidden_drop
            d_cell, d_hidden = d_state
            step_model.weights.d_initial_cells += d_cell.sum(axis=0)
            step_model.weights.d_initial_hiddens += d_hidden.sum(axis=0)
            if sgd is not None:
                for key, (weights, gradient) in updates.items():
                    sgd(weights, gradient, key=key)
            return unpad(dX)

        return outputs, recurrent_bwd

    model = wrap(recurrent_fwd, step_model)
    model.nO = step_model.nO
    return model


def RNN_step(weights, gates):
    """Create a step model for an RNN, given weights and gates functions."""

    def rnn_step_fwd(prevstate_inputs, drop=0.0):
        prevstate, inputs = prevstate_inputs
        cell_tm1, hidden_tm1 = prevstate

        acts, bp_acts = weights.begin_update((inputs, hidden_tm1), drop=drop)
        (cells, hiddens), bp_gates = gates.begin_update((acts, cell_tm1), drop=drop)

        def rnn_step_bwd(d_state_d_hiddens, sgd=None):
            (d_cells, d_hiddens), d_hiddens = d_state_d_hiddens
            d_acts, d_cell_tm1 = bp_gates((d_cells, d_hiddens), sgd=sgd)
            d_inputs, d_hidden_tm1 = bp_acts(d_acts, sgd=sgd)
            return (d_cell_tm1, d_hidden_tm1), d_inputs

        return ((cells, hiddens), hiddens), rnn_step_bwd

    model = wrap(rnn_step_fwd, weights, gates)
    model.nO = weights.nO
    model.nI = weights.nI
    model.weights = weights
    model.gates = gates
    return model


def LSTM_gates(ops):
    def lstm_gates_fwd(acts_prev_cells, drop=0.0):
        acts, prev_cells = acts_prev_cells
        new_cells = ops.allocate(prev_cells.shape)
        new_hiddens = ops.allocate(prev_cells.shape)
        ops.lstm(new_hiddens, new_cells, acts, prev_cells)
        state = (new_cells, new_hiddens)
        size = new_cells.shape[0]

        def lstm_gates_bwd(d_state, sgd=None):
            d_cells, d_hiddens = d_state
            d_cells = d_cells[:size]
            d_hiddens = d_hiddens[:size]
            d_acts = [ops.allocate(act.shape) for act in acts]
            d_prev_cells = ops.allocate(prev_cells.shape)
            ops.backprop_lstm(
                d_cells, d_prev_cells, d_acts, d_hiddens, acts, new_cells, prev_cells
            )
            return d_acts, d_prev_cells

        return state, lstm_gates_bwd

    return layerize(lstm_gates_fwd)


def _uniform_init(lo, hi):
    def wrapped(W, ops):
        copy_array(W, ops.xp.random.uniform(lo, hi, W.shape))

    return wrapped


@describe.attributes(
    nO=Dimension("Output size"),
    nI=Dimension("Input size"),
    W=Synapses(
        "Weights matrix",
        lambda obj: (obj.nO * 4, obj.nI + obj.nO),
        lambda W, ops: copy_array(W, svd_orthonormal(W.shape)),
    ),
    b=Biases("Bias vector", lambda obj: (obj.nO * 4,)),
    forget_bias=Biases(
        "Bias for forget gates",
        lambda obj: (obj.nO,),
        lambda b, ops: copy_array(b, ops.xp.ones(b.shape, dtype=b.dtype)),
    ),
    d_W=Gradient("W"),
    d_b=Gradient("b"),
    d_forget_bias=Gradient("forget_bias"),
    initial_hiddens=Biases(
        "Initial hiddens", lambda obj: (obj.nO,), _uniform_init(-0.1, 0.1)
    ),
    initial_cells=Biases(
        "Initial cells", lambda obj: (obj.nO,), _uniform_init(-0.1, 0.1)
    ),
    d_initial_hiddens=Gradient("initial_hiddens"),
    d_initial_cells=Gradient("initial_cells"),
)
class LSTM_weights(Model):
    def __init__(self, nO, nI):
        Model.__init__(self)
        self.nO = nO
        self.nI = nI

    def begin_update(self, inputs_hidden, drop=0.0):
        inputs, hidden = inputs_hidden
        assert inputs.dtype == "float32"
        X = self.ops.xp.hstack([inputs, hidden])
        acts = self.ops.gemm(X, self.W, trans2=True) + self.b
        acts = self._split_activations(acts)
        acts[0] += self.forget_bias

        def bwd_lstm_weights(d_acts, sgd=None):
            self.d_forget_bias += d_acts[0].sum(axis=0)
            d_acts = self._merge_activations(d_acts)
            dX = self.ops.gemm(d_acts, self.W)
            self.d_W += self.ops.gemm(d_acts, X, trans1=True)
            self.d_b += d_acts.sum(axis=0)
            d_input = dX[:, : self.nI]
            d_hidden = dX[:, self.nI :]
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return d_input, d_hidden

        return acts, bwd_lstm_weights

    def get_initial_state(self, n):
        initial_cells = self.ops.allocate((n, self.nO))
        initial_hiddens = self.ops.allocate((n, self.nO))
        initial_cells += self.initial_cells
        initial_hiddens += self.initial_hiddens
        return (initial_cells, initial_hiddens)

    def _split_activations(self, acts):
        acts = acts.reshape((acts.shape[0], 4, self.nO))
        acts = self.ops.xp.ascontiguousarray(acts.transpose((1, 0, 2)))
        return [acts[0], acts[1], acts[2], acts[3]]

    def _merge_activations(self, act_pieces):
        return self.ops.xp.hstack(act_pieces)
