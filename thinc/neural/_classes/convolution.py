from .model import Model
from ... import describe
from ...describe import Dimension


@describe.attributes(
    nW=Dimension("Number of surrounding tokens on each side to extract")
)
class ExtractWindow(Model):
    '''Add context to vectors in a sequence by concatenating n surrounding
    vectors.
    
    If the input is (10, 32) and n=1, the output will be (10, 96), with
    output[i] made up of (input[i-1], input[i], input[i+1]).
    '''
    name = 'extract_window'
    def __init__(self, nW=2):
        Model.__init__(self)
        self.nW = nW

    def predict(self, X):
        nr_feat = self.nW * 2 + 1
        shape = (X.shape[0], nr_feat) + X.shape[1:]
        output = self.ops.allocate(shape)
        # Let L(w[i]) LL(w[i]), R(w[i]), RR(w[i])
        # denote the words one and two to the left and right of word i.
        # So L(w[i]) == w[i-1] for i >= 1 and {empty} otherwise.
        # output[i, n-1] will be L(w[i]) so w[i-1] or empty
        # output[i, n-2] will be LL(w[i]) so w[i-2] or empty
        # etc
        for i in range(self.nW):
            output[i+1:, i] = X[:-(i+1)]
        # output[i, n] will be w[i]
        output[:, self.nW] = X
        # Now R(w[i]), RR(w[i]) etc
        for i in range(1, self.nW+1):
            output[:-i, self.nW + i] = X[:-i]
        return output.reshape(shape[0], self.ops.xp.prod(shape[1:]))

    def begin_update(self, X, drop=0.0):
        output = self.predict(X)
        output, bp_dropout = self.ops.dropout(output, drop)

        def finish_update(gradient, sgd=None, **kwargs):
            nr_feat = self.nW * 2 + 1
            shape = (gradient.shape[0], nr_feat, int(gradient.shape[-1] / nr_feat))
            gradient = gradient.reshape(shape)
            output = self.ops.allocate((gradient.shape[0], gradient.shape[-1]))
            # Word w[i+1] is the R feature of w[i]. So
            # grad(w[i+1]) += grad(R[i, n+1])
            # output[:-1] += gradient[1:, self.nW+1] 
            for i in range(1, self.nW+1): # As rightward features
                output[:-i] += gradient[i:, self.nW+i]
            # Word w[i-1] is the L feature of w[i]. So
            # grad(w[i-1]) += grad(L[i, n-1])
            # output[1:] += gradient[:-1, self.nW-1] 
            for i in range(1, self.nW+1): # As leftward features 
                output[i:] += gradient[:-i, self.nW-i]
            # Central column of gradient is the word's own feature
            output += gradient[:, self.nW, :]
            return output
        return output, bp_dropout(finish_update)
