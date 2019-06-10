from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape, layerize, wrap
from ..util import get_array_module
import numpy as np
import copy
import math


def get_qkv_self_attention(affine, nM=300, nH=6):
    nD = nM // nH
    def qkv_sa_forward(X_lengths, drop=0.0):
        X, lengths = X_lengths
        QKV, get_dX = affine.begin_update(X, drop=drop)
        Qs, Ks, Vs = _split_seqs(QKV, lengths, nH, nD)

        def qkv_sa_backward(dQs_dKs_dVs, sgd=None):
            dQs, dKs, dVs = dQs_dKs_dVs
            dQKV = _join_seqs(dQs, dKs, dVs, nH, nD)
            dX = get_dX(dQKV, sgd=sgd)
            return dX
        return (Qs, Ks, Vs), qkv_sa_backward
    return wrap(qkv_sa_forward, affine)


def _split_seqs(QKV, lengths, nH, nD):
    Qs = []
    Ks = []
    Vs = []
    i = 0
    xp = get_array_module(QKV)
    for length in lengths:
        qkv = QKV[i:i+length]
        qkv = qkv.reshape((length, 3, nH*nD))
        queries = xp.ascontiguousarray(qkv[:, 0])
        keys = xp.ascontiguousarray(qkv[:, 1])
        values = xp.ascontiguousarray(qkv[:, 2])
        Qs.append(queries.reshape((-1, nH, nD)))
        Ks.append(keys.reshape((-1, nH, nD)))
        Vs.append(values.reshape((-1, nH, nD)))
        i += length
    return Qs, Ks, Vs

def _join_seqs(Qs, Ks, Vs, nH, nD):
    xp = get_array_module(Qs[0])
    Q = xp.vstack(Qs).reshape((-1, nH*nD))
    K = xp.vstack(Ks).reshape((-1, nH*nD))
    V = xp.vstack(Vs).reshape((-1, nH*nD))
    return xp.hstack((Q, K, V))


class MultiHeadedAttention(Model):
    """Multi-headed attention. Subclasses can control the specifics by subclassing:

    * handle_inputs:
        Takes the inputs, and must return a tuple (queries, keys, values). Each
        of the elements of the tuple should be a list of arrays. The lists need
        to be the same lengths. The keys and values arrays have to be the same
        shape. The handle_inputs function needs to also return a callback,
        to do the backward pass.
    * handle_outputs:
        Performs output transforms. This makes it easier for subclasses to work
        on lists, padded batches, etc.
    * attend:
        Takes a triple (queries, keys, values) and returns a single array with
        the concatenated outputs, after applying the attention. Normally this
        will involve computing an attention matrix using (queries, keys), and then
        applying the attention matrix to values.

    The defaults for all these things are currently configured for self-attention.
    """
    def __init__(self, nM=300, nH=6):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH

    def begin_update(self, Qs_Ks_Vs, drop=0.0):
        Qs, Ks, Vs = Qs_Ks_Vs
        output, get_dQs_dKs_dVs = self.attend((Qs, Ks, Vs), drop=drop)
        return output, get_dQs_dKs_dVs

    def attend(self, Qs_Ks_Vs, drop=0.0):
        Qs, Ks, Vs = Qs_Ks_Vs
        Ys, get_dQs_dKs_dVs = self._attend_seqs(Qs, Ks, Vs)
        lengths = self.ops.asarray([len(y) for y in Ys], dtype='i')
        Y = self.ops.flatten(Ys)

        def backprop_attend(dY, sgd=None):
            dYs = self.ops.unflatten(dY, lengths)
            dQs, dKs, dVs = get_dQs_dKs_dVs(dYs, sgd=sgd)
            return dQs, dKs, dVs

        return (Y, lengths), backprop_attend

    def _attend_seqs(self, Qs, Ks, Vs):
        outputs = []
        backprops = []
        for Q, K, V in zip(Qs, Ks, Vs):
            output, backprop = self._attend(Q, K, V)
            outputs.append(output)
            backprops.append(backprop)

        def backprop_attend_seqs(d_outputs, sgd=None):
            dQs = []
            dKs = []
            dVs = []
            for d_output, backprop in zip(d_outputs, backprops):
                dQ, dK, dV = backprop(d_output, sgd=sgd)
                dQs.append(dQ)
                dKs.append(dK)
                dVs.append(dV)
            return dQs, dKs, dVs
        return outputs, backprop_attend_seqs

    def _attend(self, Q, K, V):
        """
        Compute attention on a (query, key, value) triplet.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        """
        attn, get_dQ_dK = self._get_attn_weights(Q, K)
        output, get_d_attn_dV = self._apply_attn(attn, V)

        def backprop_attend(d_output, sgd=None):
            d_attn, dV = get_d_attn_dV(d_output)
            dQ, dK = get_dQ_dK(d_attn)
            return (dQ, dK, dV)

        return output, backprop_attend

    def _get_attn_weights(self, Q0, K0):
        nQ, nK, nH, nD = (Q0.shape[0], K0.shape[0], self.nH, self.nD)
        assert Q0.shape == (nQ, nH, nD)
        assert K0.shape == (nK, nH, nD)
        sqrtM = self.ops.xp.sqrt(self.nM).astype("f")
        Q1 = _trans(Q0, 1, 0, 2)
        assert Q1.shape == (nH, nQ, nD)
        K1 = _trans(K0, 1, 2, 0)
        assert K1.shape == (nH, nD, nK)
        K1 /= sqrtM
        attn0 = self.ops.matmul(Q1, K1)
        assert attn0.shape == (nH, nQ, nK)
        attn1 = self.ops.softmax(attn0, axis=-1)
        assert attn1.shape == (nH, nQ, nK)

        def backprop_attn1(d_attn1, sgd=None):
            assert d_attn1.shape == (nH, nQ, nK)
            d_attn0 = self.ops.backprop_softmax(attn1, d_attn1, axis=-1)
            assert d_attn0.shape == (nH, nQ, nK)
            dQ1 = self.ops.matmul(d_attn0, _trans(K1, 0, 2, 1))
            assert dQ1.shape == (nH, nQ, nD)
            dK1 = self.ops.matmul(_trans(Q1, 0, 2, 1), d_attn0)
            assert dK1.shape == (nH, nD, nK)
            dK0 = _trans(dK1, 2, 0, 1)
            dK0 /= sqrtM
            assert dK0.shape == (nK, nH, nD)
            dQ0 = _trans(dQ1, 1, 0, 2)
            assert dQ0.shape == (nQ, nH, nD)
            return dQ0, dK0

        return attn1, backprop_attn1

    def _apply_attn(self, attn, V0):
        """ Multiplication with values """
        nH, nQ, nV = attn.shape
        nD = self.nD
        assert V0.shape == (nV, nH, nD)
        V1 = _trans(V0, 1, 0, 2)
        assert V1.shape == (nH, nV, nD)
        # (nH, nQ, nV) @ (nH, nV, nD) = (nH, nQ, nD)
        S0 = self.ops.matmul(attn, V1)
        assert S0.shape == (nH, nQ, nD)
        S1 = _trans(S0, 1, 0, 2)
        assert S1.shape == (nQ, nH, nD)
        S2 = S1.reshape((nQ, nH*nD))

        def backprop_apply_attn(dS2):
            assert dS2.shape == (nQ, nH*nD)
            dS1 = dS2.reshape((nQ, nH, nD))
            dS0 = dS1.transpose((1, 0, 2))
            assert dS0.shape == (nH, nQ, nD)
            dS0 = self.ops.xp.ascontiguousarray(dS0)
            # (nH, nQ, nD) @ (nH, nD, nV) --> (nH, nQ, nV)
            d_attn = self.ops.matmul(dS0, _trans(V1, 0, 2, 1))
            assert d_attn.shape == (nH, nQ, nV)
            # (nH, nV, nQ) @ (nH, nQ, nD) --> (nH, nV, nD)
            dV1 = self.ops.matmul(_trans(attn, 0, 2, 1), dS0)
            assert dV1.shape == (nH, nV, nD)
            dV0 = dV1.transpose((1, 0, 2))
            assert dV0.shape == (nV, nH, nD)
            return d_attn, dV0

        return S2, backprop_apply_attn

def _trans(X, *order):
    """Transpose and make contiguous"""
    xp = get_array_module(X)
    return xp.ascontiguousarray(X.transpose(order))
