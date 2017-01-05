import numpy


def categorical_crossentropy(scores, labels):
    target = numpy.zeros(scores.shape)
    loss = 0.
    for i in range(len(labels)):
        target[i, int(labels[i])] = 1.
        loss += (1.0-scores[i, int(labels[i])])**2
    return scores - target, loss
