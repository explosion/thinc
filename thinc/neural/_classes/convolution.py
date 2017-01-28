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
        assert gap == 0
        Model.__init__(self)
        self.nW = nW
        self.gap = gap

    def predict(self, X):
        return self.ops.seq2col(X, self.nW)

    def begin_update(self, X__bi, drop=0.0):
        X__bo = self.ops.seq2col(X__bi, self.nW)
        finish_update = self._get_finish_update()
        return X__bo, finish_update

    def _get_finish_update(self):
        def finish_update(gradient, sgd=None):
            return self.ops.backprop_seq2col(gradient, self.nW)
        return finish_update
