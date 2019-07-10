from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape, layerize, wrap
from ..util import get_array_module
import numpy as np
import copy
import math


def prepare_self_attention(affine, window=None, nM=300, nH=6):
    nD = nM // nH
    if window is not None:
        get_mask = window_mask(window)
    else:
        get_mask = None
    def qkv_sa_forward(Xs, drop=0.0):
        X = affine.ops.flatten(Xs)
        lengths = affine.ops.asarray([len(x) for x in Xs], dtype='i')
        QKV, get_dX = affine.begin_update(X, drop=drop)
        Qs, Ks, Vs = _split_seqs(QKV, lengths, nH, nD)

        def qkv_sa_backward(dQs_dKs_dVs, sgd=None):
            dQs, dKs, dVs = dQs_dKs_dVs
            dQKV = _join_seqs(dQs, dKs, dVs, nH, nD)
            dX = get_dX(dQKV, sgd=sgd)
            return affine.ops.unflatten(dX, lengths)
        if get_mask is not None:
            xp = get_array_module(lengths)
            masks = [get_mask(xp, length, length) for length in lengths]
        else:
            masks = [None for _ in lengths]
        return (Qs, Ks, Vs, masks), qkv_sa_backward
    return wrap(qkv_sa_forward, affine)


def window_mask(n):
    def get_mask(xp, nX, nY):
        mask = xp.zeros((nX, nY), dtype='f')
        for i in range(nX):
            mask[i, i-n:i+n] = 1
        return mask
    return get_mask


def _split_seqs(QKV, lengths, nH, nD):
    assert sum(lengths) == QKV.shape[0], (sum(lengths), QKV.shape[0])
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
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    return xp.hstack((Q, K, V))


class MultiHeadedAttention(Model):
    """Multi-headed attention. Requires a preprocessor to prepare (Qs, Ks, Vs, masks)
    triples, such as the prepare_self_attention() preprocessor. A layer
    should run after this to do the projection as well."""
    def __init__(self):
        Model.__init__(self)

    def begin_update(self, Qs_Ks_Vs_masks, drop=0.0):
        Qs, Ks, Vs, masks = Qs_Ks_Vs_masks
        if masks is None:
            masks = [None for _ in Qs]
        assert len(Qs) == len(Ks) == len(Vs)
        return self._attend_seqs(Qs, Ks, Vs, masks)
        
    def _attend_seqs(self, Qs, Ks, Vs, masks):
        outputs = []
        backprops = []
        assert len(Qs) == len(Ks) == len(Vs) == len(masks), (len(Qs), len(Ks), len(Vs), len(masks))
        for Q, K, V, mask in zip(Qs, Ks, Vs, masks):
            output, backprop = self._attend(Q, K, V, mask)
            outputs.append(output)
            backprops.append(backprop)
            assert output.shape[0] == Q.shape[0]

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
        assert len(outputs) == len(Qs), len(Qs)
        return outputs, backprop_attend_seqs

    def _attend(self, Q, K, V, mask):
        """
        Compute attention on a (query, key, value) triplet.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        """
        attn, get_dQ_dK = self._get_attn_weights(Q, K, mask)
        output, get_d_attn_dV = self._apply_attn(attn, V)

        def backprop_attend(d_output, sgd=None):
            d_attn, dV = get_d_attn_dV(d_output)
            dQ, dK = get_dQ_dK(d_attn)
            return (dQ, dK, dV)

        return output, backprop_attend

    def _get_attn_weights(self, Q0, K0, mask):
        nQ, nK, nH, nD = (Q0.shape[0], K0.shape[0], Q0.shape[1], Q0.shape[2])
        assert Q0.shape == (nQ, nH, nD)
        assert K0.shape == (nK, nH, nD)
        sqrtM = self.ops.xp.sqrt(nH*nD).astype("f")
        Q1 = _trans(Q0, 1, 0, 2)
        assert Q1.shape == (nH, nQ, nD)
        K1 = _trans(K0, 1, 2, 0)
        assert K1.shape == (nH, nD, nK)
        K1 /= sqrtM
        attn0 = self.ops.matmul(Q1, K1)
        assert attn0.shape == (nH, nQ, nK)
        attn1, backprop_mask = self._apply_mask(attn0, mask)
        attn2 = self.ops.softmax(attn1, axis=-1)
        assert attn2.shape == (nH, nQ, nK)

        def backprop_attn1(d_attn2, sgd=None):
            assert d_attn2.shape == (nH, nQ, nK)
            d_attn1 = self.ops.backprop_softmax(attn2, d_attn2, axis=-1)
            d_attn0 = backprop_mask(d_attn1)
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

        return attn2, backprop_attn1

    def _apply_mask(self, attn, mask):
        def backprop_apply_mask(d_attn, sgd=None):
            if mask is None:
                return d_attn
            else:
                return d_attn * mask

        if mask is None:
            return attn, backprop_apply_mask
        else:
            return attn - (1-mask)*1e9, backprop_apply_mask

    def _apply_attn(self, attn, V0):
        """ Multiplication with values """
        nH, nQ, nV = attn.shape
        nD = V0.shape[-1]
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
