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

from thinc.extra import datasets
import thinc.check
from thinc.neural.util import partition
from thinc.exceptions import ExpectedTypeError
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax, Maxout
from thinc.loss import categorical_crossentropy
from thinc.api import layerize, chain, clone, concatenate, with_flatten, Arg
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.batchnorm import BatchNorm
from thinc.neural.vecs2vec import Pooling, mean_pool, max_pool
from thinc.neural.util import remap_ids, to_categorical
from thinc.neural.ops import NumpyOps
from thinc.api import flatten_add_lengths, with_getitem
from thinc.neural._classes.spacy_vectors import SpacyVectors, get_word_ids


epoch_train_acc = 0.
def track_progress(**context):
    model = context['model']
    train_X = context['train_X']
    dev_X = context['dev_X']
    dev_y = context['dev_y']
    n_train = sum(len(x) for x in train_X)
    trainer = context['trainer']
    def each_epoch():
        global epoch_train_acc
        acc = model.evaluate(dev_X, dev_y)
        with model.use_params(trainer.optimizer.averages):
            avg_acc = model.evaluate(dev_X, dev_y)
        stats = (acc, avg_acc, float(epoch_train_acc) / n_train, trainer.dropout)
        print("%.3f (%.3f) dev acc, %.3f train acc, %.4f drop" % stats)
        epoch_train_acc = 0.
    return each_epoch


def preprocess(ops, nlp, rows):
    Xs = []
    ys = []
    for (text1, text2), label in rows:
        Xs.append((nlp(text1), nlp(text2)))
        ys.append(label)
    return Xs, to_categorical(ops.asarray(ys))


@plac.annotations(
    loc=("Location of Quora data"),
    width=("Width of the hidden layers", "option", "w", int),
    depth=("Depth of the hidden layers", "option", "d", int),
    max_batch_size=("Maximum minibatch size during training", "option", "b", int),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
)
def main(loc=None, width=64, depth=2, max_batch_size=512, dropout=0.5, dropout_decay=1e-5,
         nb_epoch=20):
    cfg = dict(locals())
    print("Load spaCy")
    nlp = spacy.load('en', parser=False, entity=False, matcher=False, tagger=False)
    print("Construct model")
    with Model.define_operators({'>>': chain, '**': clone, '|': concatenate}):
        mwe_encode = ExtractWindow(nW=1) >> Maxout(width, width*3)
        sent2vec = (
            get_word_ids
            >> flatten_add_lengths
            >> with_getitem(0,
                SpacyVectors(nlp, width)
                >> mwe_encode ** depth
            )
            >> Pooling(mean_pool, max_pool)
        )
        model = (
            ((Arg(0) >> sent2vec) | (Arg(1) >> sent2vec))
            >> Maxout(width, width*4)
            >> Maxout(width, width) ** depth
            >> Softmax(2, width)
        )

    print("Read and parse quora data")
    train, dev = datasets.quora_questions(loc)
    train_X, train_y = preprocess(model.ops, nlp, train)
    dev_X, dev_y = preprocess(model.ops, nlp, dev)
    assert len(dev_y.shape) == 2
    print("Initialize with data (LSUV)")
    with model.begin_training(train_X[:5000], train_y[:5000], **cfg) as (trainer, optimizer):
        trainer.each_epoch.append(track_progress(**locals()))
        global epoch_train_acc
        trainer.batch_size = 1
        batch_size = 1.
        print("Accuracy before training", model.evaluate(dev_X, dev_y))
        print("Train")
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            backprop(yh-y, optimizer)

            epoch_train_acc += (yh.argmax(axis=1) == y.argmax(axis=1)).sum()

            trainer.batch_size = min(int(batch_size), max_batch_size)
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
