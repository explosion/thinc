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
from thinc.neural.vecs2vec import Pooling, mean_pool, max_pool
from thinc.neural.util import remap_ids, to_categorical
from thinc.neural.ops import CupyOps


@layerize
def get_word_ids(docs, drop=0.):
    '''Get word forms.'''
    seqs = []
    for doc in docs:
        arr = numpy.zeros((len(doc)+1,), dtype='uint64')
        for token in doc:
            arr[token.i] = token.orth
        arr[len(doc)] = 0
        seqs.append(arr)
    return seqs, None


from thinc import describe
from thinc.describe import Dimension, Synapses, Gradient
from thinc.neural._lsuv import LSUVinit
@describe.on_data(LSUVinit)
@describe.attributes(
        nM=Dimension("Vector dimensions"),
        nO=Dimension("Size of output"),
        W=Synapses(
            "A projection matrix, to change vector dimensionality",
            lambda obj: (obj.nO, obj.nM),
            lambda W, ops: ops.xavier_uniform_init(W)),
        d_W=Gradient("W"),
)
class SpacyVectors(Model):
    name = 'spacy-vectors'
    def __init__(self, nlp, nO):
        Model.__init__(self)
        self._id_map = {0: 0}
        self.nO = nO
        self.nM = nlp.vocab.vectors_length
        self.nlp = nlp

    @property
    def nV(self):
        return len(self.nlp.vocab)

    def begin_update(self, ids, drop=0.):
        uniqs, inverse = self.ops.xp.unique(ids, return_inverse=True)
        vectors = self.ops.allocate((uniqs.shape[0], self.nM))
        for i, orth in enumerate(uniqs):
            vectors[i] = self.nlp.vocab[orth].vector
        def finish_update(gradients, sgd=None):
            self.d_W += self.ops.batch_outer(gradients, vectors[inverse, ])
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
            return None
        dotted = self.ops.batch_dot(vectors, self.W)
        return dotted[inverse, ], finish_update


def create_data(ops, nlp, rows):
    Xs = []
    ys = []
    for (text1, text2), label in rows:
        Xs.append((nlp(text1), nlp(text2)))
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


@layerize
def flatten_add_lengths(seqs, drop=0.):
    ops = Model.ops
    lengths = [len(seq) for seq in seqs]
    def finish_update(d_X):
        return ops.unflatten(d_X, lengths)
    X = ops.xp.concatenate(seqs)
    return (X, lengths), finish_update


def with_getitem(idx, layer):
    @layerize
    def begin_update(items, drop=0.):
        X, finish = layer.begin_update(items[idx], drop=drop)
        return items[:idx] + (X,) + items[idx+1:], finish
    return begin_update



@plac.annotations(
    loc=("Location of Quora data"),
    width=("Width of the hidden layers", "option", "w", int),
    depth=("Depth of the hidden layers", "option", "d", int),
    batch_size=("Minibatch size during training", "option", "b", int),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
)
def main(loc, width=64, depth=2, batch_size=128, dropout=0.5, dropout_decay=1e-5,
         nb_epoch=20):
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
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            backprop(yh-y, optimizer)
            #epoch_loss[-1] += loss / len(train_y)


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(100)
