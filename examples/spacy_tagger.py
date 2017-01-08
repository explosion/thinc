from __future__ import print_function, unicode_literals, division
from thinc.datasets import ewtb_pos_tags
from thinc.base import Network
from thinc.id2vec import Embed
from thinc.vec2vec import ReLu, Maxout
from thinc.vec2vec import Softmax
from thinc.convolution import ExtractWindow
from thinc.doc2vecs import SpacyWindowEncode

from thinc.util import score_model
from thinc.optimizers import linear_decay
from thinc.datasets import read_conll
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


def get_vectors(ops, docs, dropout=0.):
    docs = list(docs)
    total_words = sum(len(doc) for doc in docs)
    vectors = ops.allocate((total_words, docs[0].vocab.vector_length))
    i = 0
    for doc in docs:
        for token in doc:
            vectors[i] = token.vector
            i += 1
    return vectors, None


@toolz.curry
def parse_docs(nlp, texts, dropout=0.):
    docs = list(nlp.pipe(texts))
    return docs, None


class EncodeTagger(Model):
    width = 128
    
    def __init__(self, nr_class, width, get_vectors, **kwargs):
        self.nr_out = nr_class
        self.width = width
        Model.__init__(self, **kwargs)
        get_vectors = toolz.curry(get_vectors)(self.ops)
        self.layers = [
            MaxoutWindowEncode(width, layerize(get_vectors), name='encode'),
            ReLu(width, nr_in=width, name='relu'),
            Softmax(nr_class, nr_in=width, name='softmax')
        ]
        Model.__init__(self, *layers, **kwargs)


def spacy_conll_pos_tags(nlp, train_loc, dev_loc):
    train_sents = list(read_conll(train_loc))
    dev_sents = list(read_conll(dev_loc))
    tagmap = {}
    for words, tags, heads, labels in train_sents:
        for tag in tags:
            tagmap.setdefault(tag, len(tagmap))
    def _encode(sents):
        X = []
        y = []
        oovs = 0
        n = 0
        for words, tags, heads, labels in sents:
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
def main(nr_epoch=10, nr_sent=0):
    nlp = spacy.load('en', parser=False, tagger=False, entity=False)
    # Set shape feature
    for word in nlp.vocab:
        word.shape_ = get_word_shape(word.orth_)
    nlp.vocab.lex_attr_getters[SHAPE] = get_word_shape
    train_data, check_data, nr_class = spacy_conll_pos_tags(nlp, train_loc, dev_loc)

    if nr_sent >= 1:
        train_data = train_data[:nr_sent]
    
    model = EncodeTagger(nr_class, width, get_vectors)
    with model.begin_training(train_data) as (trainer, optimizer):
        trainer.nb_epoch = nr_epoch
        for examples, truth in trainer.iterate(model, train_data, check_data,
                                               nb_epoch=trainer.nb_epoch):
            truth = model.ops.flatten(truth)
            guess, finish_update = model.begin_update(examples, dropout=trainer.dropout)
            gradient, loss = trainer.get_gradient(guess, truth)
            optimizer.set_loss(loss)
            finish_update(gradient, optimizer)
    print("End", score_model(model, check_data))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
