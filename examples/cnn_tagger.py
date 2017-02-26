from __future__ import print_function, division
import plac
import numpy
import time
from timeit import default_timer as timer

import spacy
from spacy.attrs import ORTH, LOWER, PREFIX, SUFFIX, SHAPE
from spacy.tokens.doc import Doc

from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, Maxout, ReLu, Softmax
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.batchnorm import BatchNorm

from thinc.api import layerize, chain, concatenate, clone, add
from thinc.neural.util import flatten_sequences, remap_ids, to_categorical
from thinc.neural.ops import NumpyOps, CupyOps
from thinc.neural.optimizers import SGD

from thinc.extra.datasets import ancora_pos_tags

try:
    import cupy
except ImportError:
    print("Could not import cupy")
    cupy = None


def FeatureExtracter(lang, attrs=[LOWER, SHAPE, PREFIX, SUFFIX], tokenized=True):
    nlp = spacy.load(lang, parser=False, tagger=False,
                           entity=False, matcher=False)
    print(len(nlp.vocab))
    nlp.vocab.lex_attr_getters[PREFIX] = lambda string: string[:3]
    nlp.vocab.lex_attr_getters[SUFFIX] = lambda string: string[-3:]
    def forward(texts, drop=0.):
        if tokenized:
            docs = [Doc(nlp.vocab, words) for words in texts]
        else:
            docs = [nlp(text) for text in texts]
        features = [doc.to_array(attrs) for doc in docs]
        def backward(d_features, sgd=None):
            return d_features
        return features, backward
    return layerize(forward)


epoch_train_acc = 0.
def track_progress(**context):
    model = context['model']
    dev_X = context['dev_X']
    dev_y = model.ops.flatten(context['dev_y'])
    n_train = context['n_train']
    trainer = context['trainer']
    n_dev = len(dev_y)
    epoch_times = [timer()]
    def each_epoch():
        global epoch_train_acc
        epoch_start = epoch_times[-1]
        epoch_end = timer()
        wps_train = n_train / (epoch_end-epoch_start)
        dev_start = timer()
        acc = model.evaluate(dev_X, dev_y)
        dev_end = timer()
        wps_run = n_dev / (dev_end-dev_start)
        with model.use_params(trainer.optimizer.averages):
            avg_acc = model.evaluate(dev_X, dev_y)
        stats = (acc, avg_acc, float(epoch_train_acc) / n_train, trainer.dropout,
                 wps_train, wps_run)
        print("%.3f (%.3f) dev acc, %.3f train acc, %.4f drop, %d wps train, %d wps run" % stats)
        epoch_train_acc = 0.
        epoch_times.append(timer())
    return each_epoch


def preprocess(ops, get_feats, data, nr_tag):
    Xs, ys = zip(*data)
    Xs = [ops.asarray(x) for x in get_feats(Xs)]
    ys = [ops.asarray(to_categorical(y, nb_classes=nr_tag)) for y in ys]
    return Xs, ys


_i = 0
def debug(X, drop=0.):
    global _i
    if _i % 1000 == 0:
        print(X.mean(), X.var())
    _i += 1
    return X, lambda d, sgd: d


@plac.annotations(
    width=("Width of the hidden layers", "option", "w", int),
    vector_length=("Width of the word vectors", "option", "V", int),
    depth=("Depth of the hidden layers", "option", "d", int),
    min_batch_size=("Minimum minibatch size during training", "option", "b", int),
    max_batch_size=("Maximum minibatch size during training", "option", "B", int),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
    nb_epoch=("Maximum passes over the training data", "option", "i", int),
    L2=("L2 regularization penalty", "option", "L", float)
)
def main(width=64, depth=2, vector_length=64,
         min_batch_size=1, max_batch_size=32,
        dropout=0.9, dropout_decay=1e-3, nb_epoch=20, L2=1e-6):
    cfg = dict(locals())
    print(cfg)
    if cupy is not None:
        print("Using GPU")
        Model.ops = CupyOps()
    train_data, check_data, nr_tag = ancora_pos_tags()

    extracter = FeatureExtracter('es', attrs=[LOWER, SHAPE, PREFIX, SUFFIX])
    with Model.define_operators({'**': clone, '>>': chain, '+': add,
                                 '|': concatenate}):
        lower_case = Embed(width, vector_length, 10000, column=0)
        prefix     = Embed(width, 16, 5000, column=2)
        suffix     = Embed(width, 16, 5000, column=3)

        model = (
            layerize(flatten_sequences)
            #>> (lower_case + ((prefix | suffix) >> Maxout(width, pieces=3)))
            >> (lower_case + prefix + suffix)
            >> (ExtractWindow(nW=1) >> Maxout(width, pieces=3)) ** depth
            >> Softmax(nr_tag))

    train_X, train_y = preprocess(model.ops, extracter, train_data, nr_tag)
    dev_X, dev_y = preprocess(model.ops, extracter, check_data, nr_tag)

    n_train = float(sum(len(x) for x in train_X))
    global epoch_train_acc
    with model.begin_training(train_X, train_y, **cfg) as (trainer, optimizer):
        trainer.each_epoch.append(track_progress(**locals()))
        trainer.batch_size = min_batch_size
        batch_size = float(min_batch_size)
        for X, y in trainer.iterate(train_X, train_y):
            y = model.ops.flatten(y)

            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            loss = ((yh-y)**2).sum() / y.shape[0]
            if loss > 0.:
                optimizer.set_loss(loss)

            backprop(yh - y, optimizer)

            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001

            epoch_train_acc += (yh.argmax(axis=1) == y.argmax(axis=1)).sum()
            if epoch_train_acc / n_train >= 0.999:
                break
    with model.use_params(trainer.optimizer.averages):
        print(model.evaluate(dev_X, model.ops.flatten(dev_y)))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
