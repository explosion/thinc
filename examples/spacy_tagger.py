from __future__ import print_function, unicode_literals, division
from thinc.extra import datasets
from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, ReLu, Maxout
from thinc.neural.vec2vec import Softmax, ReLuResBN
from thinc.neural._classes.batchnorm import BatchNormalization, ScaleShift
from thinc.neural.ids2vecs import MaxoutWindowEncode
from thinc.loss import categorical_crossentropy

import numpy

from thinc.api import layerize

from thinc.neural.util import score_model
from thinc.optimizers import linear_decay
import spacy
from spacy.attrs import SHAPE
from spacy.tokens import Doc
import spacy.orth
import pathlib

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
def get_vectors(docs, dropout=0.):
    '''Given docs, return:
    Positions
    Vectors 
    Lengths

    Positions[i] will be a list of indices, showing where that word type
    occurred in a flattened list. The vector for that type will be at
    Vectors[i]. Lengths will show where the boundaries were.
    '''
    docs = list(docs)
    positions = {}
    seen = {}
    vectors = []
    for i, token in enumerate(toolz.concat(docs)):
        if token.orth not in seen:
            positions[len(seen)] = []
            seen[token.orth] = len(seen)
            if token.has_vector:
                vectors.append(token.vector / token.vector_norm)
            else:
                vectors.append(token.vector)
        positions[seen[token.orth]].append(i)
    lengths = [len(doc) for doc in docs]
    return (positions, numpy.asarray(vectors), lengths), None


@toolz.curry
def parse_docs(nlp, texts, dropout=0.):
    docs = list(nlp.pipe(texts))
    return docs, None


class EncodeTagger(Model):
    width = 128
    
    def __init__(self, nr_class, width, get_vectors, vector_dim, **kwargs):
        self.nr_out = nr_class
        self.width = width
        Model.__init__(self, **kwargs)
        self.layers = [
            MaxoutWindowEncode(width, get_vectors, nr_in=vector_dim, name='encode'),
            BatchNormalization(name='bn1'),
            ScaleShift(width, name='ss1'),
            ReLu(width, width, name='relu1'),
            BatchNormalization(name='bn1'),
            ScaleShift(width, name='ss1'),
            ReLu(width, width, name='relu2'),
            Softmax(nr_class, nr_in=width, name='softmax')
        ]

    def check_input(self, X, expect_batch=True):
        return True


def spacy_conll_pos_tags(nlp, train_sents, dev_sents):
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
            X.append(Doc(nlp.vocab, words=words))
            y.append([tagmap[tag] for tag in tags])
            oovs += sum(not w.has_vector for w in X[-1])
            n += len(X[-1])
        print(oovs, n, oovs / n)
        return zip(X, y)

    return _encode(train_sents), _encode(dev_sents), len(tagmap)


def get_word_shape(string):
    shape = spacy.orth.word_shape(string)
    if shape == 'xxxx':
        shape += string[-3:]
    return shape


@plac.annotations(
    nr_sent=("Limit number of training examples", "option", "n", int),
    nr_epoch=("Limit number of training epochs", "option", "i", int),
)
def main(nr_epoch=10, nr_sent=0, width=300):
    nlp = spacy.load('en', parser=False, tagger=False, entity=False)
    train_sents, dev_sents, _ = datasets.ewtb_pos_tags()
    train_sents, dev_sents, nr_class = spacy_conll_pos_tags(nlp, train_sents, dev_sents)

    if nr_sent >= 1:
        train_sents = train_sents[:nr_sent]
    
    model = EncodeTagger(nr_class, width, get_vectors, vector_dim=nlp.vocab.vectors_length)
    dev_X, dev_Y = zip(*dev_sents)
    dev_Y = model.ops.flatten(dev_Y)
    with model.begin_training(train_sents) as (trainer, optimizer):
        trainer.nb_epoch = nr_epoch
        trainer.dropout = 0.
        trainer.batch_size = 16
        for i in range(nr_epoch):
            for examples, truth in trainer.iterate(model, train_sents, dev_X, dev_Y,
                                                   nb_epoch=1):
                truth = model.ops.flatten(truth)
                guess, finish_update = model.begin_update(examples, dropout=trainer.dropout)
                gradient, loss = categorical_crossentropy(guess, truth)
                optimizer.set_loss(loss)
                finish_update(gradient, optimizer)
                trainer._loss += loss / len(truth)
            with model.use_params(optimizer.averages):
                print("Avg dev.: %.3f" % score_model(model, dev_X, dev_Y))
    print("End", score_model(dev_X, dev_Y))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
