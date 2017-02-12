from __future__ import print_function, unicode_literals, division
from timeit import default_timer as timer
from cytoolz import curry, concat
from thinc.extra import datasets
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Maxout, Affine
from thinc.neural.vec2vec import Softmax, Residual
from thinc.neural._classes.batchnorm import BatchNorm
from thinc.neural.ids2vecs import MaxoutWindowEncode
from thinc.neural._classes.convolution import ExtractWindow
from thinc.loss import categorical_crossentropy
from thinc.neural.optimizers import SGD
from thinc.neural.util import to_categorical
from thinc.neural._classes.spacy_vectors import SpacyVectors

from thinc.api import chain, concatenate, clone

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


#class SpacyVectors(Embed):
#    on_data_hooks = []
#    def __init__(self, nlp):
#        Model.__init__(self)
#        self._id_map = {0: 0}
#        self.nO = nlp.vocab.vectors_length
#        self.nM = self.nO
#        self.nV = len(nlp.vocab)
#        self.W.fill(0)
#        vectors = self.vectors
#        for i, word in enumerate(nlp.vocab):
#            self._id_map[word.orth] = i+1
#            vectors[i+1] = word.vector / (word.vector_norm or 1.)
#
#    def predict(self, ids):
#        return self._embed(ids)
#
#    def begin_update(self, ids, drop=0.):
#        return self.predict(ids), None
#

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


@layerize
def get_positions(ids, drop=0.):
    positions = {id_: [] for id_ in set(ids)}
    for i, id_ in enumerate(ids):
        positions[id_].append(i)
    return positions, None


@plac.annotations(
    nr_sent=("Limit number of training examples", "option", "n", int),
    nr_epoch=("Limit number of training epochs", "option", "i", int),
    dropout=("Dropout", "option", "D", float),
)
def main(nr_epoch=20, nr_sent=0, width=128, depth=3, max_batch_size=32, dropout=0.3):
    print("Loading spaCy and preprocessing")
    nlp = spacy.load('en', parser=False, tagger=False, entity=False)
    train_sents, dev_sents, _ = datasets.ewtb_pos_tags()
    train_sents, dev_sents, nr_class = spacy_preprocess(nlp, train_sents, dev_sents)
    if nr_sent >= 1:
        train_sents = train_sents[:nr_sent]

    print("Building the model")
    with Model.define_operators({'>>': chain, '|': concatenate, '**': clone}):
        model = (
            Orth
            >> SpacyVectors(nlp, width)
            >> (ExtractWindow(nW=1) >> BatchNorm(Maxout(width))) ** depth
            >> Softmax(nr_class)
        )

    print("Preparing training")
    dev_X, dev_y = zip(*dev_sents)
    dev_y = model.ops.flatten(dev_y)
    dev_y = to_categorical(dev_y, nb_classes=50)
    train_X, train_y = zip(*train_sents)
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.nb_epoch = nr_epoch
        trainer.dropout = dropout
        trainer.dropout_decay = 1e-4
        trainer.batch_size = 1
        epoch_times = [timer()]
        epoch_loss = [0.]
        n_train = sum(len(y) for y in train_y)
        def track_progress():
            start = timer()
            acc = model.evaluate(dev_X, dev_y)
            end = timer()
            with model.use_params(optimizer.averages):
                avg_acc = model.evaluate(dev_X, dev_y)
            stats = (
                epoch_loss[-1],
                acc, avg_acc,
                n_train, (end-epoch_times[-1]),
                n_train / (end-epoch_times[-1]),
                len(dev_y), (end-start),
                float(dev_y.shape[0]) / (end-start),
                trainer.dropout)
            print(
                len(epoch_loss),
                "%.3f train, %.3f (%.3f) dev, %d/%d=%d wps train, %d/%.3f=%d wps run. d.o.=%.3f" % stats)
            epoch_times.append(end)
            epoch_loss.append(0.)
        trainer.each_epoch.append(track_progress)
        print("Training")
        batch_size = 1.
        for examples, truth in trainer.iterate(train_X, train_y):
            truth = to_categorical(model.ops.flatten(truth), nb_classes=50)
            guess, finish_update = model.begin_update(examples,
                                        drop=trainer.dropout)
            n_correct = (guess.argmax(axis=1) == truth.argmax(axis=1)).sum()
            finish_update(guess-truth, optimizer)
            epoch_loss[-1] += n_correct / n_train
            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001
    with model.use_params(optimizer.averages):
        print("End: %.3f" % model.evaluate(dev_X, dev_y))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(20)
