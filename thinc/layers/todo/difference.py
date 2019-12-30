from .. import Model
from ...api import layerize
from ... import describe


def inverse(total):
    inverse = 1.0 / (1 + total)

    def backward(d_inverse):
        result = d_inverse * (-1 / (total + 1) ** 2)
        return result

    return inverse, backward


def Siamese(layer, similarity):
    def begin_update(inputs):
        input1, input2 = zip(*inputs)
        vec1, bp_vec1 = layer.begin_update(input1)
        vec2, bp_vec2 = layer.begin_update(input2)
        output, bp_output = similarity.begin_update((vec1, vec2))

        def finish_update(d_output):
            d_vec1, d_vec2 = bp_output(d_output)
            d_input1 = bp_vec1(d_vec1)
            d_input2 = bp_vec2(d_vec2)
            return (d_input1, d_input2)

        return output, finish_update

    def on_data(self, X, y):
        input1, input2 = zip(*X)
        for hook in layer.on_data_hooks:
            hook(layer, input1, y)

    model = layerize(begin_update, layers=(layer, similarity), on_data_hooks=[on_data])
    return model


def unit_init(W, ops):
    W.fill(1)


@describe.attributes(
    nO=describe.Dimension("Output size"),
    W=describe.Weights("Weights matrix", lambda obj: (obj.nO,), unit_init),
    d_W=describe.Gradient("W"),
)
class CauchySimilarity(Model):
    # From chen (2013)
    def __init__(self, length):
        Model.__init__(self)
        self.nO = length

    def begin_update(self, vec1_vec2):
        weights = self.W
        vec1, vec2 = vec1_vec2
        diff = vec1 - vec2
        square_diff = diff ** 2
        total = (weights * square_diff).sum(axis=1)
        sim, bp_sim = inverse(total)
        total = total.reshape((vec1.shape[0], 1))

        def finish_update(d_sim):
            d_total = bp_sim(d_sim)
            d_total = d_total.reshape(total.shape)
            self.d_W += (d_total * square_diff).sum(axis=0)
            d_square_diff = weights * d_total
            d_diff = 2 * d_square_diff * diff
            d_vec1 = d_diff
            d_vec2 = -d_diff
            return (d_vec1, d_vec2)

        return sim, finish_update
