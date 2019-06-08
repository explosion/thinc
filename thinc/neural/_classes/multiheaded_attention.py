from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape, layerize
from ...extra.wrappers import PyTorchWrapper
from ...extra.wrappers import xp2torch
import numpy as np
import copy
import math


try:
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    nn = None
    F = None


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
        self.get_queries = with_reshape(Affine(nM, nM))
        self.get_keys = with_reshape(Affine(nM, nM))
        self.get_values = with_reshape(Affine(nM, nM))
        self.get_output = with_reshape(Affine(nM, nM))
        self._layers = [
            self.get_queries,
            self.get_keys,
            self.get_values,
            self.get_output,
        ]
        self._softmax = PyTorchWrapper(nn.Softmax(dim=-1))
        # mask conf
        i_grad = [1, 0]
        o_xp = None
        b_map = None
        ret_x = [0]
        conf = [i_grad, o_xp, b_map, ret_x]
        self._mask = PyTorchWrapper(PytorchMaskScores(), conf=conf)

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
        S1, bp_mask = self._mask.begin_update((S0, mask))
        S2, bp_softmax = self._softmax.begin_update(S1)
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
        S0 = self.ops.xp.matmul(Q1, K2)

        # S1 shape: (nB, nH, nL, nL)
        S1 = S0.reshape((nB, nH, nL, nL))

        def backprop_attn1(dS1):
            dS0 = dS1.reshape((nB * nH, nL, nL))
            dQ1 = self.ops.xp.matmul(dS0, K2.transpose(0, 2, 1))
            dK2 = self.ops.xp.matmul(Q1.transpose(0, 2, 1), dS0)
            dK1 = (dK2 / self.ops.xp.sqrt(self.nM)).astype("float32")
            dK0 = dK1.reshape((nB, nH, nD, nL)).transpose(0, 1, 3, 2)
            dQ0 = dQ1.reshape((nB, nH, nL, nD))
            return dQ0, dK0

        return S1, backprop_attn1

    # def _mask(self, S0, mask):
    #     S1 = S0.transpose(1, 0, 2, 3)
    #     S2 = S1 - (1 - mask) * (1e9)
    #     S3 = S2.transpose(1, 0, 2, 3)
    #
    #     def backprop_attn2(dS3):
    #         dS2 = dS3.transpose(1, 0, 2, 3)
    #         dS1 = dS2
    #         dS0 = dS1.transpose(1, 0, 2, 3)
    #         return dS0
    #
    #     return S3, backprop_attn2

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
        V1 = V0.reshape((nB * nH, nL, nD))
        S1 = S0.reshape((nB * nH, nL, nL))
        S2 = self.ops.xp.matmul(S1, V1)

        S3 = S2.reshape((nB, nH, nL, nD))

        def backprop_attn4(dS3):
            dS2 = dS3.reshape((nB * nH, nL, nD))
            # (nB*nH, nL, nD) @ (nB*nH, nL, nD).T --> (nB*nH, nL, nL)
            dS1 = self.ops.xp.matmul(dS2, V1.transpose(0, 2, 1))
            # (nB*nH, nL, nL).T @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)
            dV1 = self.ops.xp.matmul(S1.transpose(0, 2, 1), dS2)
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
        self.get_queries = with_reshape(Affine(nM, nM))
        self.get_keys = with_reshape(Affine(nM, nM))
        self.get_values = with_reshape(Affine(nM, nM))
        self.get_output = with_reshape(Affine(nM, nM))
        self._layers = [
            self.get_queries,
            self.get_keys,
            self.get_values,
            self.get_output,
        ]
        self._softmax = PyTorchWrapper(nn.Softmax(dim=-1))

        # mask conf
        i_grad = [1, 0]
        o_xp = None
        b_map = None
        ret_x = [0]
        conf = [i_grad, o_xp, b_map, ret_x]
        self._mask = PyTorchWrapper(PytorchMaskScores(), conf=conf)

    def begin_update(self, input, drop=0.1):
        # Queries come from input[0], keys and values from input[1]
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
        x1, get_dq1_dk1_dv1 = self.attn(
            q1, k1, v1, mask=mask, sentX=sentX, sentY=sentY, self_attn=self_attention
        )
        x2 = x1.transpose(0, 2, 1, 3).reshape((nB, nL, nH * nD))
        x3, get_dx2 = self.get_output.begin_update(x2)

        def finish_update(dx3, sgd=None):
            dx2 = get_dx2(dx3, sgd=sgd)
            dx1 = dx2.reshape((nB, nL, nH, nD)).transpose(0, 2, 1, 3)
            dq1, dk1, dv1 = get_dq1_dk1_dv1(dx1)
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
        S1, bp_mask = self._mask.begin_update((S0, mask))
        S2, bp_softmax = self._softmax.begin_update(S1)
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
        S0 = self.ops.xp.matmul(Q1, K2)

        # S1 shape: (nB, nH, nL, nL)
        S1 = S0.reshape((nB, nH, nL, nL))

        def backprop_attn1(dS1):
            dS0 = dS1.reshape((nB * nH, nL, nL))
            dQ1 = self.ops.xp.matmul(dS0, K2.transpose(0, 2, 1))
            dK2 = self.ops.xp.matmul(Q1.transpose(0, 2, 1), dS0)
            dK1 = (dK2 / self.ops.xp.sqrt(self.nM)).astype("float32")
            dK0 = dK1.reshape((nB, nH, nD, nL)).transpose(0, 1, 3, 2)
            dQ0 = dQ1.reshape((nB, nH, nL, nD))
            return dQ0, dK0

        return S1, backprop_attn1

    # def _mask(self, S0, mask):
    #     S1 = S0.transpose(1, 0, 2, 3)
    #     S2 = S1 - (1 - mask) * (1e9)
    #     S3 = S2.transpose(1, 0, 2, 3)
    #
    #     def backprop_attn2(dS3):
    #         dS2 = dS3.transpose(1, 0, 2, 3)
    #         dS1 = dS2
    #         dS0 = dS1.transpose(1, 0, 2, 3)
    #         return dS0
    #
    #     return S3, backprop_attn2

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
        V1 = V0.reshape((nB * nH, nL, nD))
        S1 = S0.reshape((nB * nH, nL, nL))
        S2 = self.ops.xp.matmul(S1, V1)

        S3 = S2.reshape((nB, nH, nL, nD))

        def backprop_attn4(dS3):
            dS2 = dS3.reshape((nB * nH, nL, nD))
            # (nB*nH, nL, nD) @ (nB*nH, nL, nD).T --> (nB*nH, nL, nL)
            dS1 = self.ops.xp.matmul(dS2, V1.transpose(0, 2, 1))
            # (nB*nH, nL, nL).T @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)
            dV1 = self.ops.xp.matmul(S1.transpose(0, 2, 1), dS2)
            dS0 = dS1.reshape((nB, nH, nL, nL))
            dV0 = dV1.reshape((nB, nH, nL, nD))
            return dS0, dV0

        return S3, backprop_attn4


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PytorchMaskScores(nn.Module):
    def __init__(self):
        super(PytorchMaskScores, self).__init__()

    def forward(self, input):
        scores, mask = input
        mask = mask.unsqueeze(1)
        return scores.masked_fill(mask == 0, -1e9)
