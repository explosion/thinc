import math
import pdb
from .model import Model
from ...api import chain, clone, with_getitem, wrap, with_reshape
from .softmax import Softmax
from .layernorm import LayerNorm
from .resnet import Residual
from .affine import Affine
from .multiheaded_attention import MultiHeadedAttention


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
        (Y1, _), bp_self_attn = self.y_attn.begin_update((Y0, Y0, Ymask))
        # Arg0 to the multi-head attention is the queries,
        # From AIYN paper,
        # "In "encoder-decoder attention" layers, the queries come from
        # the previous decoder layer,and the memory keys and values come
        # from the output of the encoder.
        #
        # Every query (y) should be free to attend to the whole keys (X)
        (mixed, _), bp_mix_attn = self.x_attn.begin_update((Y1, X0, Xmask))
        output, bp_output = self.ffd.begin_update(mixed)

        def finish_update(dXprev_d_output, sgd=None):
            dXprev, d_output = dXprev_d_output
            d_mixed = bp_output(d_output, sgd=sgd)
            dY1, dX0 = bp_mix_attn(d_mixed, sgd=sgd)
            dY00, dY01 = bp_self_attn(dY1, sgd=sgd)
            return (dX0+dXprev, dY00+dY01)

        return ((X0, Xmask), (output, Ymask)), finish_update



