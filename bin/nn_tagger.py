"""Feed-forward neural network, using Thenao."""
import plac
import os
import sys
import time
from os import path
import shutil
import random

import numpy

import theano
import theano.tensor as T

from thinc.nn import InputLayer, EmbeddingTable

from collections import OrderedDict, defaultdict

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def read_data(pos_loc, word_map, tag_map):
    for line in open(pos_loc):
        words = []
        tags = []
        for token in line.split():
            word, tag = token.rsplit('|', 1)
            words.append(word_map[word])
            tags.append(tag_map[tag])
        yield numpy.asarray(words), numpy.asarray(tags)


class AvgParam(object):
    def __init__(self, numpy_data, name=None):
        self.curr = theano.shared(value=numpy_data, borrow=True)
        self.avg = theano.shared(value=numpy_data.copy(), borrow=True)
        self.prev_step = theano.shared(numpy.zeros(numpy_data.shape, numpy_data.dtype),
                                       borrow=True)

    def updates(self, cost, timestep, momentum=0.9, eta=0.001):
        step = (momentum * self.prev_step) - T.grad(cost, self.curr)
        curr = self.curr + (eta * step)
        alpha = (1 / timestep).clip(0.001, 0.9).astype(floatX)
        avg = ((1 - alpha) * self.avg) + (alpha * curr)
        return [(self.curr, curr), (self.prev_step, step), (self.avg, avg)]


def feed_layer(activation, weights, bias, input_):
    return activation(T.dot(input_, weights) + bias)


def L2(L2_reg, w1, w2):
    return L2_reg * ((w1 ** 2).sum() + (w2 ** 2).sum())


def sgd_sparse(param, active, cost, learning_rate):
    return [(param, T.inc_subtensor(active, -(learning_rate * T.grad(cost, active))))]


def relu(x):
    return x * (x > 0)


def build_model(n_classes, n_hidden, n_in, L1_reg, L2_reg):
    y = T.iscalar('y') 
    x = T.vector('x') 
    timestep = theano.shared(1)
    eta = theano.shared(0.001).astype(floatX)

    maxent_W, maxent_b = _init_weights(n_hidden, n_classes)
    hidden_W, hidden_b = _init_weights(n_in, n_hidden)

    # Feed the inputs forward through the network
    p_y_given_x = feed_layer(
                    T.nnet.softmax,
                    maxent_W,
                    maxent_b,
                      feed_layer(
                        relu,
                        hidden_W,
                        hidden_b,
                        x))

    cost = -T.log(p_y_given_x[0, y]) + L2(L2_reg, hidden_W, hidden_b)

    train_model = theano.function(
        name='train_model',
        inputs=[x, y],
        outputs=[-(eta * T.grad(cost, x)), T.argmax(p_y_given_x, axis=1)],
        updates=[
            (maxent_W, maxent_W - (eta * T.grad(cost, maxent_W))),
            (maxent_b, maxent_b - (eta * T.grad(cost, maxent_b))),
            (hidden_W, hidden_W - (eta * T.grad(cost, hidden_W))),
            (hidden_b, hidden_b - (eta * T.grad(cost, hidden_b))),
        ]
    )

    evaluate_model = theano.function(
        name='evaluate_model',
        inputs=[x],
        outputs=[T.argmax(p_y_given_x, axis=1)],
    )
    return train_model, evaluate_model


def normalize_embeddings(emb):
    return emb / T.sqrt((emb**2).sum(axis=1)).dimshuffle(0, 'x')


def _init_embedding(vocab_size, n_dim):
    embedding = 0.2 * numpy.random.uniform(-1.0, 1.0, (vocab_size+1, n_dim))
    return AvgParam(embedding, name='e')


def _init_weights(n_in, n_out):
    rng = numpy.random.RandomState(1234)
    weights = numpy.asarray(
        numpy.random.normal(
            loc=0.,
            scale=.0001,
            size=(n_in, n_out)),
        dtype=theano.config.floatX
    )
    bias = 0.2 * numpy.ones((n_out,), dtype=theano.config.floatX)
    return theano.shared(weights).astype(floatX), theano.shared(bias).astype(floatX)
    #return [AvgParam(weights, name='W'), AvgParam(bias, name='b')]


def iter_examples(sents):
    for words, tags in sents:
        for i in range(2, len(words)-2):
            word_slice = words[i-2:i+2]
            yield word_slice, tags[i]


def score_model(model, input_layer, examples):
    n_seen = 0
    n_true = 0
    x = numpy.ndarray((len(input_layer),), dtype='f')
    hist = numpy.asarray([0, 0])
    for word_slice, y in examples:
        input_layer.fill(x, (word_slice, hist))
        guess = model(x)
        hist[0] = hist[1]
        hist[1] = guess[0]

        n_true += guess[0] == y
        n_seen += 1
    return float(n_true) / n_seen


def main(train_loc, eval_loc):
    learning_rate = 0.001
    L1_reg = 0.00
    L2_reg = 0.0001
    n_hidden = 200
    n_epochs = 200
    random.seed(0)

    print "... reading the data"
    word_map = defaultdict(lambda: len(word_map)+1)
    tag_map = defaultdict(lambda: len(tag_map)+1)
    train_sents = list(read_data(train_loc, word_map, tag_map))
    dev_examples = list(iter_examples(read_data(eval_loc, word_map, tag_map)))

    word_vec_length = 20
    tag_vec_length = 20
    initializer = lambda i, j: 0.2 * numpy.random.uniform(-1.0, 1.0)
    input_layer = InputLayer((4, 2), 
                    [EmbeddingTable(len(word_map), word_vec_length, get_value=initializer),
                     EmbeddingTable(len(tag_map), tag_vec_length, get_value=initializer)])

    train_model, evaluate_model = build_model(len(tag_map)+1, n_hidden, len(input_layer),
                                              L1_reg, L2_reg)

    x = numpy.ndarray((len(input_layer),), dtype='f')

    print '... training'
    n_eg = 0
    examples = list(iter_examples(train_sents))[:10000]
    for epoch in range(1, n_epochs+1):
        n_seen = 0
        n_true = 0
        hist = numpy.asarray([0, 0])
        for word_slice, gold_tag in examples:
            input_layer.fill(x, (word_slice, hist))
            update, guess = train_model(x, gold_tag)
            input_layer.update(update, (word_slice, hist))

            hist[0] = hist[1]
            hist[1] = guess[0]

            n_true += guess[0] == gold_tag
            n_seen += 1
        if n_seen:
            print epoch, float(n_true) / n_seen
        random.shuffle(examples)
        if not epoch % 10:
            print 'Dev', score_model(evaluate_model, input_layer, dev_examples)



if __name__ == '__main__':
    plac.call(main)
