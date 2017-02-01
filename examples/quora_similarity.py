from __future__ import unicode_literals, print_function
import plac
import pathlib
from collections import Sequence
import numpy
import spacy
from spacy.attrs import ORTH
from spacy.tokens.doc import Doc
from preshed.maps import PreshMap
from timeit import default_timer as timer
import contextlib

from thinc.extra.datasets import read_quora_tsv_data
import thinc.check
from thinc.neural.util import partition
from thinc.exceptions import ExpectedTypeError
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax, Maxout
from thinc.loss import categorical_crossentropy
from thinc.api import layerize, chain, clone, concatenate, with_flatten, Arg
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.batchnorm import BatchNorm
from thinc.neural.vecs2vec import MultiPooling, MaxPooling, MeanPooling, MinPooling
from thinc.neural.util import remap_ids, to_categorical
from thinc.neural.ops import CupyOps
import cupy


class StaticVectors(Embed):
    def __init__(self, nlp, nO):
        self.on_init_hooks = []
        Embed.__init__(self, nO, nlp.vocab.vectors_length,
            len(nlp.vocab), is_static=True)
        vectors = self.vectors
        for i, word in enumerate(nlp.vocab):
            if word.vector_norm != 0.:
                vectors[i+1].set(word.vector)
                vectors[i+1] /= word.vector_norm


def create_data(ops, nlp, rows):
    def get_word_ids(doc):
        '''Get word forms.'''
        orths = [token.orth for token in doc]
        orths.append(0)
        return ops.asarray(orths, dtype='uint64')
    Xs = []
    ys = []
    for (text1, text2), label in rows:
        Xs.append((get_word_ids(nlp(text1)), get_word_ids(nlp(text2))))
        ys.append(label)
    return Xs, to_categorical(ops.asarray(ys))


def get_stats(model, averages, dev_X, dev_y, epoch_loss, epoch_start,
        n_train_words, n_dev_words):
    start = timer()
    acc = model.evaluate(dev_X, dev_y)
    end = timer()
    with model.use_params(averages):
        avg_acc = model.evaluate(dev_X, dev_y)
    return [
        epoch_loss, acc, avg_acc,
        n_train_words, (end-epoch_start),
        n_train_words / (end-epoch_start),
        n_dev_words, (end-start),
        float(n_dev_words) / (end-start)]

def flatten_with_lengths(layer):
    def begin_update(X, drop=0.):
        flat = layer.ops.flatten(X)
        lengths = [len(x) for x in X]
        y, bp_layer = layer.begin_update(flat, drop=drop)
        return (y, lengths), bp_layer
    return layerize(begin_update)


@plac.annotations(
    loc=("Location of Quora data"),
    width=("Width of the hidden layers", "option", "w", int),
    depth=("Depth of the hidden layers", "option", "d", int),
    batch_size=("Minibatch size during training", "option", "b", int),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
)
def main(loc, width=64, depth=2, batch_size=1, dropout=0.5, dropout_decay=1e-5,
         nb_epoch=20):
    print("Load spaCy")
    nlp = spacy.load('en', parser=False, entity=False, matcher=False, tagger=False)
    Model.ops = CupyOps()
    print("Construct model")
    with Model.define_operators({'>>': chain, '**': clone, '|': concatenate}):
        sent2vec = (
            flatten_with_lengths(
                Embed(width, nM=width, nV=10000)
                >> (ExtractWindow(nW=1) >> Maxout(width, width*3)) ** depth
            )
            >> (MeanPooling() | MaxPooling())
        )
        model = (
            ((Arg(0) >> sent2vec) | (Arg(1) >> sent2vec))
            >> Maxout(width, width*4)
            >> Maxout(width, width) ** depth
            >> Softmax(2, width)
        )

    print("Read and parse quora data")
    rows = read_quora_tsv_data(pathlib.Path(loc))
    train, dev = partition(rows, 0.9)
    train_X, train_y = create_data(model.ops, nlp, train)
    dev_X, dev_y = create_data(model.ops, nlp, dev)
    print("Train")
    with model.begin_training(train_X[:20000], train_y[:20000]) as (trainer, optimizer):
        trainer.batch_size = batch_size
        trainer.nb_epoch = nb_epoch
        trainer.dropout = dropout
        trainer.dropout_decay = dropout_decay

        epoch_times = [timer()]
        epoch_loss = [0.]
        n_train_words = sum(len(d0)+len(d1) for d0, d1 in train_X)
        n_dev_words = sum(len(d0)+len(d1) for d0, d1 in dev_X)

        def track_progress():
            stats = get_stats(model, optimizer.averages, dev_X, dev_y,
                              epoch_loss[-1], epoch_times[-1],
                              n_train_words, n_dev_words)
            stats.append(trainer.dropout)
            stats = tuple(stats)
            print(
                len(epoch_loss),
            "%.3f loss, %.3f (%.3f) acc, %d/%d=%d wps train, %d/%.3f=%d wps run. d.o.=%.3f" % stats)
            epoch_times.append(timer())
            epoch_loss.append(0.)

        trainer.each_epoch.append(track_progress)
        train_y = to_categorical(model.ops.asarray(train_y))
        batch_size = 1.
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            backprop(yh-y, optimizer)
            #epoch_loss[-1] += loss / len(train_y)
            trainer.batch_size = min(int(batch_size), 1024)
            batch_size *= 1.001


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(100)
