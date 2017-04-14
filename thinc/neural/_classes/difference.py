from thinc.api import layerize
from thinc.neural import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient


def inverse(total):
    inverse = 1. / (1+total)
    def backward(d_inverse):
        result = d_inverse * (-1 / (total+1)**2)
        return result
    return inverse, backward


def Siamese(layer, similarity):
    def begin_update(inputs, drop=0.):
        input1, input2 = zip(*inputs)
        vec1, bp_vec1 = layer.begin_update(input1, drop=drop)
        vec2, bp_vec2 = layer.begin_update(input2, drop=drop)
        output, bp_output = similarity.begin_update((vec1, vec2), drop=drop)
        def finish_update(d_output, sgd=None):
            d_vec1, d_vec2 = bp_output(d_output, sgd)
            d_input1 = bp_vec1(d_vec1, sgd)
            d_input2 = bp_vec2(d_vec2, sgd)
            return (d_input1, d_input2)
        return output, finish_update
    model = layerize(begin_update)

    model._layers.append(layer)
    model._layers.append(similarity)
    def on_data(self, X, y):
        input1, input2 = zip(*X)
        for hook in layer.on_data_hooks:
            hook(layer, input1, y)
    model.on_data_hooks.append(on_data)
    return model

def unit_init(W, ops):
    W.fill(1)

@describe.attributes(
    nO=Dimension("Output size"),
    W=Synapses("Weights matrix", lambda obj: (obj.nO,), unit_init),
    d_W=Gradient("W")
)
class CauchySimilarity(Model):
    # From chen (2013)
    def __init__(self, length):
        Model.__init__(self)
        self.nO = length

    def begin_update(self, vec1_vec2, drop=0.):
        weights = self.W
        vec1, vec2 = vec1_vec2
        diff = vec1-vec2
        dropout_mask = self.ops.get_dropout_mask(diff.shape, drop)
        square_diff = diff ** 2
        if dropout_mask is not None:
            square_diff *= dropout_mask
        total = (weights * square_diff).sum(axis=1)
        sim, bp_sim = inverse(total)
        total = total.reshape((vec1.shape[0], 1))
        def finish_update(d_sim, sgd=None):
            d_total = bp_sim(d_sim)
            d_total = d_total.reshape(total.shape)
            self.d_W += (d_total * square_diff).sum(axis=0)
            d_square_diff = weights * d_total
            if dropout_mask is not None:
                d_square_diff *= dropout_mask
            d_diff = 2 * d_square_diff * diff
            d_vec1 = d_diff
            d_vec2 = -d_diff
            sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return (d_vec1, d_vec2)
        return sim, finish_update
