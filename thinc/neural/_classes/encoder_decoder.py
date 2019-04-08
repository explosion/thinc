import math
import pdb
from .model import Model
from ...api import chain, clone, with_getitem, wrap, with_reshape
from .softmax import Softmax
from .layernorm import LayerNorm
from .resnet import Residual
from .affine import Affine
from .multiheaded_attention import MultiHeadedAttention


def with_reshape(layer):
    def with_reshape_forward(X, drop=0.):
        initial_shape = X.shape
        final_shape = list(initial_shape[:-1]) + [layer.nO]
        nB = X.shape[0]
        nL = X.shape[1]
        X2d = X.reshape(-1, X.shape[-1])
        X2d = X2d.astype(layer.ops.xp.float32)
        Y2d, Y2d_backprop = layer.begin_update(X2d, drop=drop)
        Y = Y2d.reshape(final_shape)

        def with_reshape_backward(dY, sgd=None):
            dY2d = dY.reshape(nB*nL, -1).astype(layer.ops.xp.float32)
            return Y2d_backprop(dY2d, sgd=sgd).reshape(initial_shape)
        return Y, with_reshape_backward
    return wrap(with_reshape_forward, layer)


class EncoderDecoder(Model):
    def __init__(self, nS=1, nH=6, nM=300, nTGT=10000):
        '''
        EncoderDecoder consists of an encoder stack, a decoder stack and an
        output layer which is a linear + softmax.
        Parameters explanation:
            nS: the number of encoders/decoders in the stack
            nH: the number of heads in the multiheaded attention
            nM: the token's embedding size
            nTGT: the number of unique words in output vocabulary
        '''
        Model.__init__(self)
        self.nS = nS
        self.nH = nH
        self.nM = nM
        self.nTGT = nTGT
        self.enc = clone(EncoderLayer(self.nH, self.nM), 1)
        # self.dec = clone(DecoderLayer(self.nH, self.nM), 1)
        self.dec = PoolingDecoder(self.nM)
        self.proj = with_reshape(Softmax(nO=nTGT, nI=nM))
        self._layers = [self.enc, self.dec, self.proj]

    def begin_update(self, inputs, drop=0.0):
        '''
        A batch object flows through the network. It contains input, output and
        corresponding masks. Input changes while the object travels through
        the network. Output is the golden output.
        Input: nB x nL x nM
        '''
        if len(inputs) == 2:
            (X0, Xmask), (Y0, Ymask) = inputs
            sentX = None
            sentY = None
        else:
            (X0, Xmask), (Y0, Ymask), (sentX, sentY) = inputs
        (X1, _), backprop_encode = self.enc.begin_update((X0, Xmask, sentX), drop=drop)
        (_, (Y1, _)), backprop_decode = self.dec.begin_update(((X1, Xmask, sentX), (Y0, Ymask, sentY)), drop=drop)
        word_probs, backprop_output = self.proj.begin_update(Y1, drop=drop)
        # Right-shift the word probabilities
        word_probs[:, 1:] = word_probs[:, :-1]
        word_probs[:, 0] = 0

        def finish_update(d_word_probs, sgd=None):
            # Unshift
            d_word_probs[:, :-1] = d_word_probs[:, 1:]
            d_word_probs[:, -1] = 0.

            dY1 = backprop_output(d_word_probs, sgd=sgd)
            zeros = Model.ops.xp.zeros(X0.shape, dtype=Model.ops.xp.float32)
            dX1, dY0 = backprop_decode((zeros, dY1), sgd=sgd)
            dX0 = backprop_encode(dX1, sgd=sgd)
            return (dX0, dY0)

        return (word_probs, Xmask), finish_update


def EncoderLayer(nH, nM):
    return chain(
        MultiHeadedAttention(nM, nH, layer='Encoder'),
        with_getitem(0, with_reshape(LayerNorm(Affine(nM, nM))))
    )


class PoolingDecoder(Model):
    def __init__(self, nM):
        Model.__init__(self)
        self.nM = nM
        self.ffd = LayerNorm(Maxout(nO=nM, nI=nM*3, pieces=3))
        self._layers = [self.ffd]

    def begin_update(self, X_Y, drop=0.):
        (X0, Xmask, _), (Y0, Ymask, _) = X_Y
        X_masked = self.ops.xp.copy(X0)
        X_masked[Xmask[:, 0, :] == 0] = -math.inf
        Xpool = X_masked.max(axis=1, keepdims=True)

        Y_masked = self.ops.xp.copy(Y0)
        Y_masked[Ymask[:, -1, :] == 0] = -math.inf
        Ypool = self.ops.allocate((X0.shape[0], X0.shape[1], self.nM))

        # maxing only over previous elements
        for i in range(X0.shape[1]):
            Ypool[:, i, :] = Y_masked[:, :i+1, :].max(axis=1, keepdims=True).squeeze()

        mixed = self.ops.allocate((X0.shape[0], X0.shape[1], self.nM*3))
        mixed[:, :, :self.nM] = Xpool
        mixed[:, :, self.nM:self.nM*2] = Ypool
        mixed[:, :, self.nM*2:] = Y0
        output, bp_output = self.ffd.begin_update(mixed.reshape((-1, self.nM*3)))
        output = output.reshape((X0.shape[0], X0.shape[1], output.shape[1]))

        def backprop_pooling_decoder(dX_d_output, sgd=None):
            dXin, d_output = dX_d_output
            d_output = d_output.reshape((X0.shape[0]*X0.shape[1], d_output.shape[2]))
            d_mixed = bp_output(d_output, sgd=sgd)
            d_mixed = d_mixed.reshape((X0.shape[0], X0.shape[1], d_mixed.shape[1]))
            dXpool = d_mixed[:, :, :self.nM]
            dYpool = d_mixed[:, :, self.nM:self.nM*2]
            dY0 = d_mixed[:, :, self.nM*2:]
            dX0 = self.ops.allocate(X0.shape)
            for i in range(X0.shape[0]):
                for j in range(X0.shape[1]):
                    for k in range(X0.shape[2]):
                        if X0[i, j, k] >= Xpool[i, 0, k]:
                            dX0[i, j, k] += dXpool[i, j, k]
            for i in range(Y0.shape[0]):
                for j in range(Y0.shape[1]):
                    for k in range(Y0.shape[2]):
                        if Y0[i, j, k] >= Ypool[i, j, k]:
                            dY0[i, j, k] += dYpool[i, j, k]
            return dXin + dX0, dY0
        return ((X0, Xmask), (output, Ymask)), backprop_pooling_decoder


class DecoderLayer(Model):
    def __init__(self, nH, nM):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM
        self.x_attn = MultiHeadedAttention(nM, nH, layer='Decoder')
        self.y_attn = MultiHeadedAttention(nM, nH, layer='Decoder')
        self.ffd = with_reshape(LayerNorm(Affine(nM, nM)))
        self._layers = [self.x_attn, self.y_attn, self.ffd]

    def begin_update(self, X_Y, drop=0.0):
        (X0, Xmask, sentX), (Y0, Ymask, sentY) = X_Y
        (Y1, _), bp_self_attn = self.y_attn.begin_update((Y0, Ymask, sentY))
        (mixed, _), bp_mix_attn = self.x_attn.begin_update((Y1, X0, Xmask, sentY, sentX))
        output, bp_output = self.ffd.begin_update(mixed)

        def finish_update(dXprev_d_output, sgd=None):
            dXprev, d_output = dXprev_d_output
            d_mixed = bp_output(d_output, sgd=sgd)
            dY1, dX0 = bp_mix_attn(d_mixed, sgd=sgd)
            dY0 = bp_self_attn(dY1, sgd=sgd)
            return (dX0 + dXprev, dY0)

        return ((X0, Xmask), (output, Ymask)), finish_update
