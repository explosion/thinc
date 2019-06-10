from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape, layerize
from ..util import get_array_module
import numpy as np
import copy
import math


class SparseAttention(Model):
    """This class implements multiheaded attention in steps, factorizing
    the attention matrix."""

    def __init__(self, nM=300, nH=6):
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH
        self.get_queries = Affine(nM, nM)
        self.get_keys = Affine(nM, nM)
        self.get_values = Affine(nM, nM)
        self.get_output = Affine(nM, nM)
        self._layers = [
            self.get_queries,
            self.get_keys,
            self.get_values,
            self.get_output,
        ]

    def begin_update(self, input, drop=0.1):
        if len(input) == 3:
            x0, mask, sentX = input
            sentY = sentX
            y0 = x0
            self_attention = True
        else:
            self_attention = False
            x0, y0, mask, sentX, sentY = input
        """ Shapes """
        # x0: nB, nL, nM
        # q0: nB, nL, nM
        # k0: nB, nL, nM
        # v0: nB, nL, nM
        # q1: nB, nH, nL, nD
        # k1: nB, nH, nL, nD
        # v1: nB, nH, nL, nD
        nB, nL, nD, nH = x0.shape[0], x0.shape[1], self.nD, self.nH
        q0, get_dx0 = self.get_queries.begin_update(x0)
        q1 = q0.reshape(nB, -1, self.nH, self.nD).transpose(0, 2, 1, 3)
        k0, get_dy0_1 = self.get_keys.begin_update(y0)
        k1 = k0.reshape(nB, -1, self.nH, self.nD).transpose(0, 2, 1, 3)
        v0, get_dy0_2 = self.get_values.begin_update(y0)
        v1 = v0.reshape(nB, -1, self.nH, self.nD).transpose(0, 2, 1, 3)
        mask1 = mask.astype(Model.ops.xp.uint8) & self.mask_floor(nB, nL)
        mask2 = mask.astype(Model.ops.xp.uint8) & self.mask_repetitive(nB, nL)
        q2, get_dq1_dk1_dv1 = self.attn(
            q1, k1, v1, mask=mask1, sentX=sentX, sentY=sentY, self_attn=self_attention
        )
        x1, get_dq2_dk1_dv1 = self.attn(
            q1, k1, v1, mask=mask2, sentX=sentX, sentY=sentY, self_attn=self_attention
        )

        x2 = x1.transpose(0, 2, 1, 3).reshape((nB, nL, nH * nD))
        x3, get_dx2 = self.get_output.begin_update(x2)

        def finish_update(dx3, sgd=None):
            dx2 = get_dx2(dx3, sgd=sgd)
            dx1 = dx2.reshape((nB, nL, nH, nD)).transpose(0, 2, 1, 3)
            dq2, dk11, dv11 = get_dq2_dk1_dv1(dx1)
            dq1, dk12, dv12 = get_dq1_dk1_dv1(dq2)
            dk1 = dk11 + dk12
            dv1 = dv11 + dv12
            nM = nH * nD
            dq0 = dq1.transpose(0, 2, 1, 3).reshape((nB, nL, nM))
            dk0 = dk1.transpose(0, 2, 1, 3).reshape((nB, nL, nM))
            dv0 = dv1.transpose(0, 2, 1, 3).reshape((nB, nL, nM))
            dy0 = get_dy0_2(dv0, sgd=sgd) + get_dy0_1(dk0, sgd=sgd)
            dx0 = get_dx0(dq0, sgd=sgd)
            if self_attention:
                return dx0 + dy0
            else:
                return (dx0, dy0)

        return x3, finish_update

    def attn(self, Q, K, V, mask=None, sentX=None, sentY=None, self_attn=True):
        """
        Compute attention on (query, key, value) triplets.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        """
        S0, bp_scaled_dp = self._scaled_dot_prod(Q, K)
        S1, bp_mask = self._mask(S0, mask)
        S2, bp_softmax = self._softmax(S1)
        S3, bp_apply_attn = self._apply_attn(S2, V)

        def backprop_attn(dS3):
            """ Attention three inputs, one output """
            dS2, dV = bp_apply_attn(dS3)
            dS1 = bp_softmax(dS2)
            dS0 = bp_mask(dS1)
            dQ, dK = bp_scaled_dp(dS0)
            return dQ, dK, dV

        return S3, backprop_attn

    def _scaled_dot_prod(self, Q0, K0):
        # Q0: nB, nH, nL, nD
        # K0: nB, nH, nL, nD
        nB, nH, nL, nD = Q0.shape
        # Q1: nB*nH, nL, nD
        Q1 = Q0.reshape((nB * nH, nL, nD))
        # K1: (nB*nH, nD, nL)
        K1 = K0.transpose(0, 1, 3, 2).reshape((nB * nH, nD, nL))
        # K2: (nB*nH, nD, nL)
        K2 = (K1 / self.ops.xp.sqrt(self.nM)).astype("float32")
        # S0: (nB*nH, nL, nL)
        S0 = self.ops.matmul(self.ops.xp.ascontiguousarray(Q1), self.ops.xp.ascontiguousarray(K2))

        # S1 shape: (nB, nH, nL, nL)
        S1 = S0.reshape((nB, nH, nL, nL))

        def backprop_attn1(dS1):
            dS0 = self.ops.xp.ascontiguousarray(dS1.reshape((nB * nH, nL, nL)))
            dQ1 = self.ops.matmul(dS0, self.ops.xp.ascontiguousarray(K2.transpose(0, 2, 1)))
            dK2 = self.ops.matmul(self.ops.xp.ascontiguousarray(Q1.transpose(0, 2, 1)), dS0)
            dK1 = (dK2 / self.ops.xp.sqrt(self.nM)).astype("float32")
            dK0 = dK1.reshape((nB, nH, nD, nL)).transpose(0, 1, 3, 2)
            dQ0 = dQ1.reshape((nB, nH, nL, nD))
            return dQ0, dK0

        return S1, backprop_attn1
    
    def _softmax(self, X, drop=0.):
        Y = self.ops.softmax(X, axis=-1)
        def backprop_softmax(dY, sgd=None):
            return self.ops.backprop_softmax(Y, dY, axis=-1)
        return Y, backprop_softmax

    def _mask(self, S0, mask):
        mask = self.ops.asarray(mask, dtype='f')
        S1 = S0.transpose(1, 0, 2, 3)
        S2 = S1 - (1 - mask) * (1e9)
        S3 = S2.transpose(1, 0, 2, 3)
        def backprop_attn2(dS3):
            dS2 = dS3.transpose(1, 0, 2, 3)
            dS1 = dS2 * mask
            dS0 = dS1.transpose(1, 0, 2, 3)
            return dS0
        return S3, backprop_attn2

    def _apply_attn(self, S0, V0):
        """ Multiplication with values """
        # S0: (nB, nH, nL, nL)
        # VO: (nB, nH, nL, nD)
        # S1: (nB*nH, nL, nL)
        # V1:  (nB*nH, nL, nD)
        # S2: (nB*nH, nL, nD)
        # S3: (nB, nH, nL, nD)
        nB, nH, nL, nL = S0.shape
        nD = V0.shape[-1]
        V1 = self.ops.xp.ascontiguousarray(V0.reshape((nB * nH, nL, nD)))
        S1 = self.ops.xp.ascontiguousarray(S0.reshape((nB * nH, nL, nL)))
        S2 = self.ops.matmul(S1, V1)

        S3 = S2.reshape((nB, nH, nL, nD))

        def backprop_attn4(dS3):
            dS2 = self.ops.xp.ascontiguousarray(dS3.reshape((nB * nH, nL, nD)))
            # (nB*nH, nL, nD) @ (nB*nH, nL, nD).T --> (nB*nH, nL, nL)
            dS1 = self.ops.matmul(dS2, self.ops.xp.ascontiguousarray(V1.transpose(0, 2, 1)))
            # (nB*nH, nL, nL).T @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)
            dV1 = self.ops.matmul(self.ops.xp.ascontiguousarray(S1.transpose(0, 2, 1)), dS2)
            dS0 = dS1.reshape((nB, nH, nL, nL))
            dV0 = dV1.reshape((nB, nH, nL, nD))
            return dS0, dV0

        return S3, backprop_attn4

    def mask_floor(self, nB, nL):
        stride = math.ceil(math.sqrt(nL))
        floor_mask = (
            Model.ops.xp.expand_dims(Model.ops.xp.eye(nL), axis=0)
            .repeat(nB, axis=0)
            .astype(Model.ops.xp.uint8)
        )
        for i in range(nL):
            lower = max(0, i - (i % stride))
            higher = i + 1
            floor_mask[:, i, lower:higher] = 1
        return floor_mask

    def mask_repetitive(self, nB, nL, c=1, mode="left"):
        """ Every stride tokens, mask one (independent of row) """
        stride = math.ceil(math.sqrt(nL))
        repetitive_mask = (
            Model.ops.xp.expand_dims(Model.ops.xp.eye(nL), axis=0)
            .repeat(nB, axis=0)
            .astype(Model.ops.xp.uint8)
        )
        for j in range(nL):
            if (j % stride) >= (stride - c):
                if mode == "left":
                    repetitive_mask[:, j:, j] = 1
        return repetitive_mask


class MultiHeadedAttention(Model):
    """This class implements multiheaded attention. It can be used for self
    attention or outer attention, depending on our needs. There is no left
    and right context width. We attend to the whole sentence and we take
    care of the masks to adjust appropriately. There are no actual different
    weight matrices for each head, but a bigger weight matrix for all heads.
    Going to bigger dimensions is the key to get the multiple heads.
    For the time being; key, query and value matrices are supposed to have the
    same length.
    """
    def __init__(self, nM=300, nH=6):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH
        self.get_queries_keys_values = Affine(nM*3, nM)
        self.get_output = Affine(nM, nM)
        self._layers = [
            self.get_queries_keys_values,
            self.get_output,
        ]

    def begin_update(self, X_lengths, drop=0.0):
        X, lengths = X_lengths
        QKV, get_dX = self.get_queries_keys_values.begin_update(X, drop=drop)
        Qs, Ks, Vs = self._split_seqs(QKV, lengths)
        Xattns, get_dQs_dKs_dVs = self._attend_seqs(Qs, Ks, Vs)
        assert Xattns[0].shape == (lengths[0], X.shape[1]), (Xattns[0].shape, X.shape[1])
        Xattn = self.ops.flatten(Xattns)
        assert Xattn.shape == X.shape
        Y, get_dXattn = self.get_output.begin_update(Xattn, drop=drop)

        def backprop_self_attn(dY, sgd=None):
            dXattn = get_dXattn(dY, sgd=sgd)
            dXattns = self.ops.unflatten(dXattn, lengths)
            dQs, dKs, dVs = get_dQs_dKs_dVs(dXattns, sgd=sgd)
            dQKV = self._join_seqs(dQs, dKs, dVs)
            dX = get_dX(dQKV, sgd=sgd)
            return dX
        
        return (Y, lengths), backprop_self_attn

    def _split_seqs(self, QKV, lengths):
        Qs = []
        Ks = []
        Vs = []
        i = 0
        for length in lengths:
            qkv = QKV[i:i+length]
            qkv = qkv.reshape((length, 3, self.nM))
            queries = self.ops.xp.ascontiguousarray(qkv[:, 0])
            keys = self.ops.xp.ascontiguousarray(qkv[:, 1])
            values = self.ops.xp.ascontiguousarray(qkv[:, 2])
            Qs.append(queries.reshape((-1, self.nH, self.nD)))
            Ks.append(keys.reshape((-1, self.nH, self.nD)))
            Vs.append(values.reshape((-1, self.nH, self.nD)))
            i += length
        return Qs, Ks, Vs

    def _join_seqs(self, Qs, Ks, Vs):
        Q = self.ops.xp.vstack(Qs).reshape((-1, self.nH*self.nD))
        K = self.ops.xp.vstack(Ks).reshape((-1, self.nH*self.nD))
        V = self.ops.xp.vstack(Vs).reshape((-1, self.nH*self.nD))
        return self.ops.xp.hstack((Q, K, V))

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
