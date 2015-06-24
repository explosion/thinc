"""Feed-forward neural network, using Thenao."""
import plac
import os
import sys
import time
from os import path
import shutil
import random

import numpy
from collections import OrderedDict, defaultdict

from thinc.nn import InputLayer, EmbeddingTable
from thinc.theano_nn import compile_theano_model


def read_data(pos_loc, word_map, tag_map):
    for line in open(pos_loc):
        words = []
        tags = []
        for token in line.split():
            word, tag = token.rsplit('|', 1)
            words.append(word_map[word])
            tags.append(tag_map[tag])
        yield numpy.asarray(words), numpy.asarray(tags)


def normalize_embeddings(emb):
    return emb / T.sqrt((emb**2).sum(axis=1)).dimshuffle(0, 'x')


def _init_embedding(vocab_size, n_dim):
    embedding = 0.2 * numpy.random.uniform(-1.0, 1.0, (vocab_size+1, n_dim))
    return AvgParam(embedding, name='e')


def iter_examples(sents):
    for words, tags in sents:
        words = [0, 0] + list(words) + [0, 0]
        tags = [0, 0] + list(tags) + [0, 0]
        for i in range(2, len(words) - 2):
            yield words[i-2:i+3], tags[i]


def score_model(model, input_layer, examples):
    n_seen = 0
    n_true = 0
    x = numpy.ndarray((len(input_layer),), dtype='f')
    hist = numpy.asarray([0, 0])
    for word_slice, y in examples:
        input_layer.fill(x, (word_slice, hist), use_avg=True)
        guess = model(x)
        hist[0] = hist[1]
        hist[1] = guess[0]

        n_true += guess[0] == y
        n_seen += 1
    return float(n_true) / n_seen


def main(train_loc, eval_loc):
    eta = 0.001
    mu = 0.9
    L1_reg = 0.0000
    L2_reg = 0.0000
    n_hidden = 10
    n_epochs = 2000
    random.seed(0)

    print "... reading the data"
    word_map = defaultdict(lambda: len(word_map)+1)
    tag_map = defaultdict(lambda: len(tag_map)+1)
    train_sents = list(read_data(train_loc, word_map, tag_map))
    dev_examples = list(iter_examples(read_data(eval_loc, word_map, tag_map)))

    word_vec_length = 10
    tag_vec_length = 10
    initializer = lambda: 0.2 * numpy.random.uniform(-1.0, 1.0)
    input_layer = InputLayer((5, 2), 
                    [EmbeddingTable(len(word_map), word_vec_length, initializer),
                     EmbeddingTable(len(tag_map), tag_vec_length, initializer)])

    print len(input_layer)

    train_model, evaluate_model = compile_theano_model(
                                    len(tag_map)+1,
                                    n_hidden,
                                    len(input_layer),
                                    L1_reg,
                                    L2_reg)

    x = numpy.ndarray((len(input_layer),), dtype='f')

    print '... training'
    n_eg = 0
    examples = list(iter_examples(train_sents))
    for epoch in range(1, n_epochs+1):
        n_seen = 0
        n_true = 0
        hist = numpy.asarray([0, 0])
        for word_slice, gold_tag in examples:
            input_layer.fill(x, (word_slice, hist), use_avg=False)
            probs, update, guess = train_model(x, gold_tag, eta)
            input_layer.update(update, (word_slice, hist), n_seen+1, eta, mu)

            hist[0] = hist[1]
            hist[1] = guess[0]

            n_true += guess[0] == gold_tag
            n_seen += 1
        if n_seen:
            print epoch, float(n_true) / n_seen
        random.shuffle(examples)
        eta *= 0.96
        if not epoch % 10:
            print 'Dev', score_model(evaluate_model, input_layer, dev_examples)



if __name__ == '__main__':
    plac.call(main)
