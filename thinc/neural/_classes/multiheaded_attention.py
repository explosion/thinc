from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape, layerize
import numpy as np
import copy
import math



def with_pad_and_mask(layer):
    def create_model_input_forward(Xs, drop=0.):
        nX = model.ops.asarray([x.shape[0] for x in Xs], dtype='i')
        nL = nX.max()
        X, unpad_X = pad_sequences(model.ops, Xs, pad_to=nL)
        X_mask = get_mask(X, nX)
        Y, bp_Y = layer.begin_update((X.astype("float32"), X_mask, None), drop=drop)
        def create_model_input_backward(dYs, sgd=None):
            dY, _ = pad_sequences(model.ops, dYs, pad_to=nL)
            dX = bp_Y(dY, sgd=sgd)
            return unpad_X(dX)
        return unpad_X(Y), create_model_input_backward
    model = layerize(create_model_input_forward)
    return model


def pad_sequences(ops, seqs_in, pad_to=None):
    lengths = ops.asarray([len(seq) for seq in seqs_in], dtype='i')
    nB = len(seqs_in)
    if pad_to is None:
        pad_to = lengths.max()
    arr = ops.allocate((nB, int(pad_to)) + seqs_in[0].shape[1:], dtype=seqs_in[0].dtype)
    for arr_i, seq in enumerate(seqs_in):
        arr[arr_i, :seq.shape[0]] = ops.asarray(seq)
    
    def unpad(padded):
        unpadded = [None] * len(lengths)
        for i in range(padded.shape[0]):
            unpadded[i] = padded[i, :lengths[i]]
        return unpadded
    return arr, unpad


def get_mask(X, nX):
    nB = X.shape[0]
    nL = X.shape[1]
    X_mask = Model.ops.allocate((nB, nL, nL))
    for i, length in enumerate(nX):
        X_mask[i, :, :length] = 1.0
    return X_mask


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
        Xattn, get_dQKV = self.attn(QKV, lengths)
        Y, get_dXattn = self.get_output.begin_update(Xattn, drop=drop)

        def backprop_self_attn(dY, sgd=None):
            dXattn = get_dXattn(dY, sgd=sgd)
            dQKV = get_dQKV(dXattn, sgd=sgd)
            dX = get_dX(dQKV, sgd=sgd)
            return dX
        
        return (Y, lengths), backprop_self_attn

    def attn(self, QKV, lengths):
        """
        Compute attention on (query, key, value) triplets.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        """
        outputs = []
        backprops = []
        i = 0
        for length in lengths:
            Q = QKV[i:i+length, :self.nM]
            K = QKV[i:i+length, self.nM:self.nM*2]
            V = QKV[i:i+length, -self.nM:]
            S0, bp_scaled_dp = self._scaled_dot_prod(Q, K)
            S1, bp_softmax = self._softmax(S0)
            S2, bp_apply_attn = self._apply_attn(S1, V)
            backprops.append((bp_scaled_dp, bp_softmax, bp_apply_attn))
            outputs.append(S2)
            i += length

        def backprop_attn(d_outputs, sgd=None):
            """ Attention three inputs, one output """
            i = 0
            d_inputs = []
            for length, (bp_scaled_dp, bp_softmax, bp_apply_attn) in zip(lengths, backprops):
                dS2 = d_outputs[i:i+length]
                dS1, dV = bp_apply_attn(dS2)
                dS0 = bp_softmax(dS1)
                dQ, dK = bp_scaled_dp(dS0)
                assert dQ.shape == dK.shape == dV.shape
                dQKV = self.ops.xp.hstack((dQ, dK, dV))
                d_inputs.append(dQKV)
                i += length
            d_inputs = self.ops.xp.vstack(d_inputs)
            return d_inputs.reshape(QKV.shape)
        return self.ops.xp.vstack(outputs), backprop_attn

    def _scaled_dot_prod(self, Q0, K0):
        Q0 = Q0.reshape((-1, self.nH, self.nD)).transpose((1, 0, 2))
        K0 = K0.reshape((-1, self.nH, self.nD)).transpose((1, 0, 2))
        # Q0: nH, nL, nD
        # K0: nH, nL, nD
        nH, nL, nD = Q0.shape
        # Q1: nH, nL, nD
        Q1 = Q0.reshape((nH, nL, nD))
        # K1: (nH, nD, nL)
        K1 = K0.transpose(0, 2, 1).reshape((nH, nD, nL))
        # K2: (nH, nD, nL)
        K2 = (K1 / self.ops.xp.sqrt(self.nM)).astype("float32")
        # S0: (nH, nL, nL)
        S0 = self.ops.matmul(self.ops.xp.ascontiguousarray(Q1), self.ops.xp.ascontiguousarray(K2))

        def backprop_attn1(dS0, sgd=None):
            dQ1 = self.ops.matmul(dS0, self.ops.xp.ascontiguousarray(K2.transpose(0, 2, 1)))
            dK2 = self.ops.matmul(self.ops.xp.ascontiguousarray(Q1.transpose(0, 2, 1)), dS0)
            dK1 = (dK2 / self.ops.xp.sqrt(self.nM)).astype("float32")
            dK0 = dK1.reshape((nH, nD, nL)).transpose(2, 0, 1)
            dQ0 = dQ1.reshape((nH, nL, nD)).transpose((1, 0, 2))
            return dQ0, dK0

        return S0, backprop_attn1

    def _softmax(self, X, drop=0.):
        Y = self.ops.softmax(X, axis=-1)
        def backprop_softmax(dY, sgd=None):
            return self.ops.backprop_softmax(Y, dY, axis=-1)
        return Y, backprop_softmax

    def _apply_attn(self, S0, V0):
        """ Multiplication with values """
        # S0: (nH, nL, nL)
        # VO: (nL, nH, nD)
        # S1: (nH, nL, nD)
        # V1:  (nH, nL, nD)
        # S2: (nL, nH*nD)
        nH, nL, nL = S0.shape
        V0 = V0.reshape((nL, nH, -1))
        nD = V0.shape[-1]
        V1 = V0.transpose((1, 0, 2))
        S1 = self.ops.matmul(self.ops.xp.ascontiguousarray(S0), self.ops.xp.ascontiguousarray(V1))
        S2 = S1.transpose((1, 0, 2)).reshape((-1, self.nH*self.nD))

        def backprop_attn4(dS2):
            dS1 = dS2.reshape((-1, self.nH, self.nD)).transpose((1, 0, 2))
            dS1 = self.ops.xp.ascontiguousarray(dS1)
            # (nH, nL, nD) @ (nH, nL, nD).T --> (nH, nL, nL)
            dS0 = self.ops.matmul(dS1, self.ops.xp.ascontiguousarray(V1.transpose(0, 2, 1)))
            # (nH, nL, nL).T @ (nH, nL, nD) --> (nH, nL, nD)
            dV1 = self.ops.matmul(self.ops.xp.ascontiguousarray(S0.transpose(0, 2, 1)), dS1)
            dV0 = dV1.reshape((nH, nL, nD)).transpose((1, 0, 2))
            return dS0, dV0

        return S2, backprop_attn4
