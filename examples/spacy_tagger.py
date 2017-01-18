from __future__ import print_function, unicode_literals, division
from cytoolz import curry
from thinc.extra import datasets
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Maxout
from thinc.neural.vec2vec import Softmax, Residual
from thinc.neural._classes.batchnorm import BatchNorm
from thinc.neural.ids2vecs import MaxoutWindowEncode
from thinc.neural._classes.convolution import ExtractWindow
from thinc.loss import categorical_crossentropy
from thinc.neural.optimizers import SGD

from thinc.api import chain, concatenate

import numpy

from thinc.api import layerize

from thinc.neural.optimizers import linear_decay
import spacy
from spacy.attrs import SHAPE
from spacy.tokens import Doc
from spacy.strings import StringStore
import spacy.orth
import pathlib
import numpy.random
import numpy.linalg

import plac

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import cytoolz as toolz
except ImportError:
    import toolz

@layerize
def Orth(docs, drop=0.):
    '''Get word forms.'''
    ids = numpy.zeros((sum(len(doc) for doc in docs),), dtype='i')
    i = 0
    for doc in docs:
        for token in doc:
            ids[i] = token.orth
            i += 1
    return ids, None


@layerize
def Shape(docs, drop=0.):
    '''Get word shapes.'''
    ids = numpy.zeros((sum(len(doc) for doc in docs),), dtype='i')
    i = 0
    for doc in docs:
        for token in doc:
            ids[i] = token.shape
            i += 1
    return ids, None


@layerize
def Prefix(docs, drop=0.):
    '''Get prefixes.'''
    ids = numpy.zeros((sum(len(doc) for doc in docs),), dtype='i')
    i = 0
    for doc in docs:
        for token in doc:
            ids[i] = token.prefix
            i += 1
    return ids, None


@layerize
def Suffix(docs, drop=0.):
    '''Get suffixes.'''
    ids = numpy.zeros((sum(len(doc) for doc in docs),), dtype='i')
    i = 0
    for doc in docs:
        for token in doc:
            ids[i] = token.suffix
            i += 1
    return ids, None


def spacy_preprocess(nlp, train_sents, dev_sents):
    tagmap = {}
    for words, tags in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
    def _encode(sents):
        X = []
        y = []
        oovs = 0
        n = 0
        for words, tags in sents:
            for word in words:
                _ = nlp.vocab[word]
            X.append(Doc(nlp.vocab, words=words))
            y.append([tagmap[tag] for tag in tags])
            oovs += sum(not w.has_vector for w in X[-1])
            n += len(X[-1])
        print(oovs, n, oovs / n)
        return zip(X, y)
    return _encode(train_sents), _encode(dev_sents), len(tagmap)


@plac.annotations(
    nr_sent=("Limit number of training examples", "option", "n", int),
    nr_epoch=("Limit number of training epochs", "option", "i", int),
)
def main(nr_epoch=20, nr_sent=0, width=128):
    print("Loading spaCy and preprocessing")
    nlp = spacy.load('en', parser=False, tagger=False, entity=False)
    train_sents, dev_sents, _ = datasets.ewtb_pos_tags()
    train_sents, dev_sents, nr_class = spacy_preprocess(nlp,
            train_sents, dev_sents)
    if nr_sent >= 1:
        train_sents = train_sents[:nr_sent]
    
    print("Building the model")
    with Model.define_operators({'>>': chain, '|': concatenate}):
        features = (
            (Orth     >> Embed(32, 32, len(nlp.vocab.strings)))
            | (Shape  >> Embed(8, 8, len(nlp.vocab.strings)))
            | (Prefix >> Embed(8, 8, len(nlp.vocab.strings)))
            | (Suffix >> Embed(8, 8, len(nlp.vocab.strings)))
        )
        model = (
            features
            >> ExtractWindow(nW=2)
            >> BatchNorm(ReLu(width))
            >> BatchNorm(ReLu(width))
            >> ExtractWindow(nW=2)
            >> BatchNorm(ReLu(width))
            >> Softmax(nr_class)
        )

    print("Preparing training")
    dev_X, dev_Y = zip(*dev_sents)
    dev_Y = model.ops.flatten(dev_Y)
    train_X, train_y = zip(*train_sents)
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.nb_epoch = nr_epoch
        trainer.dropout = 0.0
        trainer.dropout_decay = 1e-5
        trainer.batch_size = 8
        epoch_loss = [0] # Workaround Python 2 scoping
        def log_progress():
            with model.use_params(optimizer.averages):
                progress = (model.evaluate(dev_X, dev_Y), epoch_loss[0])
                print("Avg dev.: %.3f, loss %.3f" % progress)
            epoch_loss[0] = 0

        trainer.each_epoch.append(log_progress)
        print("Training")
        for examples, truth in trainer.iterate(train_X, train_y):
            truth = model.ops.flatten(truth)
            guess, finish_update = model.begin_update(examples,
                                        drop=trainer.dropout)
            gradient, loss = categorical_crossentropy(guess, truth)
            optimizer.set_loss(loss)
            finish_update(gradient, optimizer)
            trainer._loss += loss / len(truth)
            epoch_loss[0] += loss
    with model.use_params(optimizer.averages):
        print("End: %.3f" % model.evaluate(dev_X, dev_Y))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
