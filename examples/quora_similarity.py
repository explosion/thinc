from __future__ import unicode_literals, print_function
import plac
import pathlib
from collections import Sequence
import csv
import numpy
import spacy
from spacy.attrs import ORTH
from spacy.tokens.doc import Doc
from preshed.maps import PreshMap
from timeit import default_timer as timer
import contextlib

import thinc.check
from thinc.ops import NumpyOps
from thinc.exceptions import ExpectedTypeError
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Softmax, Maxout
from thinc.loss import categorical_crossentropy
from thinc.api import layerize, chain, clone, concatenate
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.batchnorm import BatchNorm
from thinc.neural.vecs2vec import MultiPooling, MaxPooling, MeanPooling, MinPooling


def is_docs(arg_id, args, kwargs):
    docs = args[arg_id]
    if not isinstance(docs, Sequence):
        raise ExpectedTypeError(type(docs), ['Sequence'])
    if not isinstance(docs[0], Doc):
        raise ExpectedTypeError(type(docs[0]), ['spacy.tokens.doc.Doc'])


def read_quora_tsv_data(loc):
    is_header = True
    with loc.open('rb') as file_:
        for row in csv.reader(file_, delimiter=b'\t'):
            if is_header:
                is_header = False
                continue
            id_, qid1, qid2, sent1, sent2, is_duplicate = row
            sent1 = sent1.decode('utf8').strip()
            sent2 = sent2.decode('utf8').strip()
            if sent1 and sent2:
                yield (sent1, sent2), int(is_duplicate)


def create_data(nlp, rows):
    Xs = []
    ys = []
    for (text1, text2), label in rows:
        Xs.append((nlp(text1), nlp(text2)))
        ys.append(label)
    return Xs, ys


def partition(examples, split_size): # pragma: no cover
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


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


def with_flatten(layer):
    def begin_update(seqs_in, drop=0.):
        lengths = [len(seq) for seq in seqs_in]
        X, bp_layer = layer.begin_update(layer.ops.flatten(seqs_in), drop=drop)
        if bp_layer is None:
            return layer.ops.unflatten(X, lengths), None

        def finish_update(d_seqs_out, sgd=None):
            d_X = bp_layer(layer.ops.flatten(d_seqs_out), sgd=sgd)
            return layer.ops.unflatten(d_X, lengths) if d_X is not None else None
        return layer.ops.unflatten(X, lengths), finish_update
    model = layerize(begin_update)
    model._layers.append(layer)
    return model


class StaticVectors(Embed):
    def __init__(self, nlp, nO):
        Embed.__init__(self,
            nO,
            nlp.vocab.vectors_length,
            len(nlp.vocab),
            is_static=True)
        vectors = self.vectors
        for i, word in enumerate(nlp.vocab):
            self._id_map[word.orth] = i+1
            vectors[i+1] = word.vector / (word.vector_norm or 1.)

def Arg(i):
    @layerize
    def begin_update(batched_inputs, drop=0.):
        inputs = zip(*batched_inputs)
        return inputs[i], None
    return begin_update


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

@plac.annotations(
    loc=("Location of Quora data"),
    width=("Width of the hidden layers", "option", "w", int),
    depth=("Depth of the hidden layers", "option", "d", int),
)
def main(loc, width=64, depth=2):
    print("Load spaCy")
    nlp = spacy.load('en', parser=False, entity=False, matcher=False, tagger=False)
    with Model.define_operators({'>>': chain, '**': clone, '|': concatenate}):
        sent2vec = (
            get_word_ids
            >> with_flatten(
                 StaticVectors(nlp, width)
                 >> (ExtractWindow(nW=1) >> Maxout(width, width*3)) ** depth)
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
    train_X, train_y = create_data(nlp, train)
    dev_X, dev_y = create_data(nlp, dev)
    print("Train")
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.batch_size = 128
        trainer.nb_epoch = 10
        trainer.dropout = 0.0
        trainer.dropout_decay = 1e-4
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
            d_loss, loss = categorical_crossentropy(yh, y)
            optimizer.set_loss(loss)
            backprop(d_loss, optimizer)
            epoch_loss[-1] += loss / len(train_y)


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(100)
