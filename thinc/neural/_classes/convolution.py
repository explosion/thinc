from .model import Model
from ... import describe
from ...describe import Dimension, AttributeDescription


@describe.attributes(
    nW=Dimension("Number of surrounding tokens on each side to extract"),
    gap=AttributeDescription("Number of nearest tokens to skip, to offset the window")
)
class ExtractWindow(Model):
    '''Add context to vectors in a sequence by concatenating n surrounding
    vectors.
    
    If the input is (10, 32) and n=1, the output will be (10, 96), with
    output[i] made up of (input[i-1], input[i], input[i+1]).
    '''
    name = 'extract_window'
    def __init__(self, nW=2, gap=0):
        Model.__init__(self)
        self.nW = nW
        self.gap = gap

    def predict(self, X):
        nr_feat = self.nW * 2 + 1
        shape = (X.shape[0], nr_feat) + X.shape[1:]
        output = self.ops.allocate(shape)
        for i in range(1, self.nW+1):
            output[:-(self.gap+i), self.nW-i] = X[self.gap+i:]
        # output[i, n] will be w[i]
        output[:, self.nW] = X
        for i in range(1, self.nW+1):
            output[self.gap+i:, self.nW + i] = X[:-(self.gap+i)]
        return output.reshape(shape[0], self.ops.xp.prod(shape[1:]))

    def begin_update(self, X__bi, drop=0.0):
        X__bo = self.predict(X__bi)
        X__bo, bp_dropout = self.ops.dropout(X__bo, drop)
        finish_update = self._get_finish_update()
        return X__bo, bp_dropout(finish_update)
    
    def _get_finish_update(self):
        return lambda d, sgd=None: backprop_concatenate(self.ops, d, self.nW, self.gap)


def backprop_concatenate(ops, dY__bo, nW, gap):
    nr_feat = nW * 2 + 1
    bfi = (dY__bo.shape[0], nr_feat, int(dY__bo.shape[-1] / nr_feat))
    dY__bfi = dY__bo.reshape(bfi)
    dX__bi = ops.allocate((dY__bo.shape[0], bfi[-1]))
    for f in range(1, nW+1):
        dX__bi[gap+f:] += dY__bfi[:-(gap+f), nW-f] # Words at start not used as rightward feats
    dX__bi += dY__bfi[:, nW]
    for f in range(1, nW+1):
        dX__bi[:-(gap+f)] += dY__bfi[(gap+f):, nW+f] # Words at end not used as leftward feats
    return dX__bi
