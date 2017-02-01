import numpy

try:
    from cupy import get_array_module
except ImportError:
    def get_array_module(*a, **k):
        return numpy


def categorical_crossentropy(scores, labels):
    xp = get_array_module(scores)
    target = xp.zeros(scores.shape, dtype='float32')
    loss = 0.
    for i in range(len(labels)):
        target[i, int(labels[i])] = 1.
        loss += (1.0-scores[i, int(labels[i])])**2
    return scores - target, loss
