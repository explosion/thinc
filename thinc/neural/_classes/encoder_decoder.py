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
        self.enc = clone(EncoderLayer(self.nH, self.nM), self.nS)
        self.dec = clone(DecoderLayer(self.nH, self.nM), self.nS)
        self.proj = with_reshape(Softmax(nO=nTGT, nI=nM))
        self._layers = [self.enc, self.dec, self.proj]

    def begin_update(self, inputs, drop=0.0):
        '''
        A batch object flows through the network. It contains input, output and
        corresponding masks. Input changes while the object travels through
        the network. Output is the golden output.
        Input: nB x nL x nM
        '''
        (X0, Xmask), (Y0, Ymask) = inputs
        # b0: x0, y0
        # b1: x1, y1
        # b2: x2, y2
        (X1, _), backprop_encode = self.enc.begin_update((X0, Xmask), drop=drop)
        (_, (Y1, _)), backprop_decode = self.dec.begin_update(((X1, Xmask), (Y0, Ymask)), drop=drop)
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
        MultiHeadedAttention(nM, nH),
        with_getitem(0, with_reshape(LayerNorm(Residual(Affine(nM, nM)))))
    )


class DecoderLayer(Model):
    def __init__(self, nH, nM):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM
        self.x_attn = MultiHeadedAttention(nM, nH)
        self.y_attn = MultiHeadedAttention(nM, nH)
        self.ffd = with_reshape(LayerNorm(Residual(Affine(nM, nM))))
        self._layers = [self.x_attn, self.y_attn, self.ffd]

    def begin_update(self, X_Y, drop=0.0):
        (X0, Xmask), (Y0, Ymask) = X_Y
        (Y1, _), get_dY00_dY01 = self.y_attn.begin_update((Y0, Y0, Ymask))
        (Y2, _), get_dY1_dX0 = self.x_attn.begin_update((Y1, X0, Xmask))
        Y3, get_dY2 = self.ffd.begin_update(Y2)

        def finish_update(dY3_dX0, sgd=None):
            dY3, dX = dY3_dX0
            dY2 = get_dY2(dY3, sgd=sgd)
            dY1, dX0 = get_dY1_dX0(dY2, sgd=sgd)
            dY00, dY01 = get_dY00_dY01(dY1, sgd=sgd)
            dY0 = dY00 + dY01
            dX += dX0
            return (dX, dY0,)

        return ((X0, Xmask), (Y3, Ymask)), finish_update
