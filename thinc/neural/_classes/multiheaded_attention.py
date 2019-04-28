from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape
from thinc.extra.visualizer import visualize_attention
from thinc.extra.wrappers import xp2torch
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch
from thinc.extra.wrappers import PyTorchWrapper
import math


class MultiHeadedAttention(Model):
    ''' This class implements multiheaded attention. It can be used for self
    attention or outer attention, depending on our needs. There is no left
    and right context width. We attend to the whole sentence and we take
    care of the masks to adjust appropriately. There are no actual different
    weight matrices for each head, but a bigger weight matrix for all heads.
    Going to bigger dimensions is the key to get the multiple heads.
    For the time being; key, query and value matrices are supposed to have the
    same length.
    '''
    def __init__(self, nM=300, nH=6, layer='Encoder'):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH
        self.layer = layer
        self.get_queries = with_reshape(Affine(nM, nM))
        self.get_keys = with_reshape(Affine(nM, nM))
        self.get_values = with_reshape(Affine(nM, nM))
        self.get_output = with_reshape(Affine(nM, nM))
        self._layers = [self.get_queries, self.get_keys, self.get_values, self.get_output]
        self._softmax = PyTorchWrapper(nn.Softmax(dim=-1))

        ''' mask conf '''
        i_grad = [1, 0]
        o_xp = None
        b_map = None
        ret_x = [0]
        conf = [i_grad, o_xp, b_map, ret_x]
        self._mask = PyTorchWrapper(PytorchMaskScores(), conf=conf)

    def begin_update(self, input, drop=0.0):
        # TESTED
        # Queries come from input[0], keys and values from input[1]
        if len(input) == 3:
            x0, mask, sentX = input
            sentY = sentX
            y0 = x0
            self_attention = True
        else:
            self_attention = False
            x0, y0, mask, sentX, sentY = input
        ''' Shapes '''
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
        x1, get_dq1_dk1_dv1 = self.attn(q1, k1, v1, mask=mask, sentX=sentX,
                                        sentY=sentY, self_attn=self_attention)
        x2 = x1.transpose(0, 2, 1, 3).reshape((nB, nL, nH*nD))
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
        '''
        Compute attention on (query, key, value) triplets.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        '''
        # TESTED
        S0, bp_scaled_dp = self._scaled_dot_prod(Q, K)
        S1, bp_mask = self._mask.begin_update((S0, mask))
        S2, bp_softmax = self._softmax.begin_update(S1)
        S3, bp_apply_attn = self._apply_attn(S2, V)

        def backprop_attn(dS3):
            ''' Attention three inputs, one output '''
            dS2, dV = bp_apply_attn(dS3)
            dS1 = bp_softmax(dS2)
            dS0 = bp_mask(dS1)
            dQ, dK = bp_scaled_dp(dS0)
            return dQ, dK, dV
        return S3, backprop_attn

    def _scaled_dot_prod(self, Q0, K0):
        # TESTED
        # Q0: nB, nH, nL, nD
        # K0: nB, nH, nL, nD
        nB, nH, nL, nD = Q0.shape
        # Q1: nB*nH, nL, nD
        Q1 = Q0.reshape((nB*nH, nL, nD))
        # K1: (nB*nH, nD, nL)
        K1 = K0.transpose(0, 1, 3, 2).reshape((nB*nH, nD, nL))
        # K2: (nB*nH, nD, nL)
        K2 = (K1 / self.ops.xp.sqrt(self.nM)).astype("float32")
        # S0: (nB*nH, nL, nL)
        S0 = self.ops.xp.matmul(Q1, K2)

        # S1 shape: (nB, nH, nL, nL)
        S1 = S0.reshape((nB, nH, nL, nL))

        def backprop_attn1(dS1):
            dS0 = dS1.reshape((nB*nH, nL, nL))
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
        ''' Multiplication with values '''
        # TESTED
        # S0: (nB, nH, nL, nL)
        # VO: (nB, nH, nL, nD)
        # S1: (nB*nH, nL, nL)
        # V1:  (nB*nH, nL, nD)
        # S2: (nB*nH, nL, nD)
        # S3: (nB, nH, nL, nD)
        nB, nH, nL, nL = S0.shape
        nD = V0.shape[-1]
        V1 = V0.reshape((nB*nH, nL, nD))
        S1 = S0.reshape((nB*nH, nL, nL))
        S2 = self.ops.xp.matmul(S1, V1)

        S3 = S2.reshape((nB, nH, nL, nD))

        def backprop_attn4(dS3):
            dS2 = dS3.reshape((nB*nH, nL, nD))
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
