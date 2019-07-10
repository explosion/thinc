# coding: utf8
from __future__ import unicode_literals, print_function, division

import plac
from timeit import default_timer as timer
from srsly import cloudpickle as pickle

import spacy
from spacy.attrs import LOWER, PREFIX, SUFFIX, SHAPE
from spacy.tokens.doc import Doc

from thinc.i2v import HashEmbed
from thinc.v2v import Model, Affine, Maxout, Softmax
from thinc.t2t import ExtractWindow
from thinc.neural._classes.multiheaded_attention import MultiHeadedAttention
from thinc.neural._classes.multiheaded_attention import prepare_self_attention
from thinc.misc import Residual, LayerNorm
from thinc.api import with_flatten, flatten_add_lengths, unflatten, with_getitem

from thinc.api import layerize, chain, concatenate, clone, add
from thinc.neural.util import to_categorical, prefer_gpu
from thinc.extra.datasets import ancora_pos_tags

# from thinc.api import FeatureExtracter

try:
    import cupy
except ImportError:
    print("Could not import cupy")
    cupy = None


def FeatureExtracter(lang, attrs=[LOWER, SHAPE, PREFIX, SUFFIX], tokenized=True):
    nlp = spacy.blank(lang)
    nlp.vocab.lex_attr_getters[PREFIX] = lambda string: string[:3]
    nlp.vocab.lex_attr_getters[SUFFIX] = lambda string: string[-3:]

    def forward(texts, drop=0.0):
        if tokenized:
            docs = [Doc(nlp.vocab, words) for words in texts]
        else:
            docs = [nlp(text) for text in texts]
        features = [doc.to_array(attrs) for doc in docs]

        def backward(d_features, sgd=None):
            return d_features

        return features, backward

    return layerize(forward)


def PositionEncode(L, D):
    positions = Model.ops.position_encode(L, D)
    def position_encode_forward(Xs, drop=0.):
        output = []
        for x in Xs:
            output.append(x + positions[:x.shape[0]])
        def position_encode_backward(dYs, sgd=None):
            return dYs
        return output, position_encode_backward
    return layerize(position_encode_forward)


epoch_train_acc = 0.0


def track_progress(**context):
    model = context["model"]
    dev_X = context["dev_X"]
    dev_y = model.ops.flatten(context["dev_y"])
    n_train = context["n_train"]
    trainer = context["trainer"]
    n_dev = len(dev_y)
    epoch_times = [timer()]
    losses = context["losses"]

    def each_epoch():
        global epoch_train_acc
        epoch_start = epoch_times[-1]
        epoch_end = timer()
        wps_train = n_train / (epoch_end - epoch_start)
        dev_start = timer()
        acc = model.evaluate(dev_X, dev_y)
        dev_end = timer()
        wps_run = n_dev / (dev_end - dev_start)
        with model.use_params(trainer.optimizer.averages):
            avg_acc = model.evaluate(dev_X, dev_y)
        stats = (
            acc,
            avg_acc,
            float(losses[-1]),
            trainer.dropout,
            wps_train,
            wps_run,
        )
        print(
            "%.3f (%.3f) dev acc, %.3f train loss, %.4f drop, %d wps train, %d wps run"
            % stats
        )
        epoch_train_acc = 0.0
        epoch_times.append(timer())
        losses.append(0.)

    return each_epoch


def preprocess(ops, get_feats, data, nr_tag, npad=4):
    Xs, ys = zip(*data)
    Xs = [ops.asarray(x) for x in get_feats(Xs)]
    ys = [ops.asarray(to_categorical(y, nb_classes=nr_tag)) for y in ys]
    return Xs, ys


_i = 0


def debug(X, drop=0.0):
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
    learn_rate=("Learning rate", "option", "e", float),
    momentum=("Momentum", "option", "m", float),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
    nb_epoch=("Maximum passes over the training data", "option", "i", int),
    L2=("L2 regularization penalty", "option", "L", float),
)
def main(
    width=128,
    depth=4,
    vector_length=64,
    min_batch_size=1,
    max_batch_size=32,
    learn_rate=0.001,
    momentum=0.9,
    dropout=0.5,
    dropout_decay=1e-4,
    nb_epoch=20,
    L2=1e-6,
):
    cfg = dict(locals())
    print(cfg)
    prefer_gpu()
    train_data, check_data, nr_tag = ancora_pos_tags()

    extracter = FeatureExtracter("es", attrs=[LOWER, SHAPE, PREFIX, SUFFIX])
    Model.lsuv = True
    with Model.define_operators({"**": clone, ">>": chain, "+": add, "|": concatenate}):
        lower_case = HashEmbed(width, 100, column=0)
        shape = HashEmbed(width // 2, 200, column=1)
        prefix = HashEmbed(width // 2, 100, column=2)
        suffix = HashEmbed(width // 2, 100, column=3)

        model = (
            with_flatten(
                (lower_case | shape | prefix | suffix)
                >> Maxout(width, width+(width//2)*3, pieces=3))
            >> PositionEncode(1000, width)
            >> Residual(
                prepare_self_attention(Affine(width*3, width), nM=width, nH=4)
                >> MultiHeadedAttention()
                >> with_flatten(Affine(width, width)))
            >> with_flatten(Softmax(nr_tag, width))
        )

    train_X, train_y = preprocess(model.ops, extracter, train_data, nr_tag)
    dev_X, dev_y = preprocess(model.ops, extracter, check_data, nr_tag)

    n_train = float(sum(len(x) for x in train_X))
    global epoch_train_acc
    losses = [0.]
    with model.begin_training(train_X[:5000], train_y[:5000], **cfg) as (
        trainer,
        optimizer,
    ):
        trainer.each_epoch.append(track_progress(**locals()))
        trainer.batch_size = min_batch_size
        batch_size = float(min_batch_size)
        trainer.dropout = 0.1
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)

            gradient = [yh[i] - y[i] for i in range(len(yh))]
            losses[-1] += sum((g**2).sum() for g in gradient)

            backprop(gradient, optimizer)

            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001
    with model.use_params(trainer.optimizer.averages):
        print(model.evaluate(dev_X, model.ops.flatten(dev_y)))
        with open("/tmp/model.pickle", "wb") as file_:
            pickle.dump(model, file_)


if __name__ == "__main__":
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats

        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
