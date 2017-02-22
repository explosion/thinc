from __future__ import division
from thinc.api import layerize
from thinc.neural import Model


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


# From chen (2013)
def CauchySimilarity(ops, length):
    weights = ops.allocate(length)
    weights += 1.
    Model.id += 1
    id_ = Model.id
    def begin_update(vec1_vec2, drop=0.):
        vec1, vec2 = vec1_vec2
        diff = vec1-vec2
        square_diff = diff ** 2
        total = (weights * square_diff).sum(axis=1)
        sim, bp_sim = inverse(total)
        total = total.reshape((vec1.shape[0], 1))
        def finish_update(d_sim, sgd=None):
            d_total = ops.asarray(bp_sim(d_sim), dtype='float32')
            d_total = d_total.reshape(total.shape)
            d_weights = (d_total * square_diff).sum(axis=0)
            d_square_diff = weights * d_total
            d_diff = 2 * d_square_diff * diff
            d_vec1 = d_diff
            d_vec2 = -d_diff
            sgd(weights.ravel(), d_weights.ravel(), key=id_)
            return (d_vec1, d_vec2)
        return sim, finish_update
    return layerize(begin_update)
