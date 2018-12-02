# coding: utf8
from __future__ import unicode_literals, print_function, division

import sys
import plac
import numpy
from timeit import default_timer as timer
from pathlib import Path

from spacy.attrs import ORTH, LOWER, PREFIX, SUFFIX, SHAPE
from spacy.tokens.doc import Doc
from thinc.extra.load_nlp import get_spacy

from thinc.neural._classes.hash_embed import HashEmbed
from thinc.neural._classes.static_vectors import StaticVectors
from thinc.neural.vec2vec import Model, Maxout, Softmax
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural._classes.batchnorm import BatchNorm as BN

from thinc.api import layerize, chain, concatenate, clone, add
from thinc.neural.util import flatten_sequences, to_categorical
from thinc.neural.ops import CupyOps


try:
    import cupy
except ImportError:
    print("Could not import cupy")
    cupy = None


def twitter_ner():
    loc = Path("/home/rd/repos/twitter_nlp/data/annotated/wnut16/data/")
    tagmap = {}
    train_X, train_y = _read_conll_ner((loc / "train").open(), tagmap)
    dev_X, dev_y = _read_conll_ner((loc / "dev").open(), tagmap)
    print(tagmap)
    return zip(train_X, train_y), zip(dev_X, dev_y), tagmap


def print_dev_sentences(model, orig_words, gold_tags, coded_words, tag_map):
    reverse_tag_map = {id_: tag for tag, id_ in tag_map.items()}
    scores = model(coded_words)
    i = 0
    for sent_words, sent_gold in zip(orig_words, gold_tags):
        for word, gold in zip(sent_words, sent_gold):
            tag = scores[i].argmax()
            print(
                "%s\t%s\t%s"
                % (word, reverse_tag_map[int(gold)], reverse_tag_map[int(tag)])
            )
            i += 1
        print()


def _read_conll_ner(file_, tagmap):
    Xs = [[]]
    ys = [[]]
    for line in file_:
        if not line.strip():
            if Xs[-1] and ys[-1]:
                Xs.append([])
                ys.append([])
        else:
            word, tag = line.strip().split()
            Xs[-1].append(word)
            ys[-1].append(tagmap.setdefault(tag, len(tagmap)))
    if not Xs[-1]:
        Xs.pop()
        ys.pop()
    ys = [numpy.asarray(y, dtype="int32") for y in ys]
    return Xs, ys


def FeatureExtracter(lang, attrs=[LOWER, SHAPE, PREFIX, SUFFIX], tokenized=True):
    nlp = get_spacy(lang, parser=False, tagger=False, entity=False)
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


def Residual(layer):
    def forward(X, drop=0.0):
        y, bp_y = layer.begin_update(X, drop=drop)
        output = X + y

        def backward(d_output, sgd=None):
            return d_output + bp_y(d_output, sgd)

        return output, backward

    model = layerize(forward)
    model._layers.append(layer)

    def on_data(self, X, y=None):
        for layer in self._layers:
            for hook in layer.on_data_hooks:
                hook(layer, X, y)

    model.on_data_hooks.append(on_data)
    return model


epoch_train_acc = 0.0


def track_progress(**context):
    model = context["model"]
    dev_X = context["dev_X"]
    dev_y = model.ops.flatten(context["dev_y"])
    n_train = context["n_train"]
    trainer = context["trainer"]
    n_dev = len(dev_y)
    epoch_times = [timer()]

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
            float(epoch_train_acc) / n_train,
            trainer.dropout,
            wps_train,
            wps_run,
        )
        print(
            "%.3f (%.3f) dev acc, %.3f train acc, %.4f drop, %d wps train, %d wps run"
            % stats,
            file=sys.stderr,
        )
        epoch_train_acc = 0.0
        epoch_times.append(timer())

    return each_epoch


def preprocess(ops, get_feats, data, nr_tag):
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
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
    nb_epoch=("Maximum passes over the training data", "option", "i", int),
    L2=("L2 regularization penalty", "option", "L", float),
    device=("Device", "option", "G", str),
)
def main(
    width=300,
    depth=4,
    vector_length=64,
    min_batch_size=1,
    max_batch_size=32,
    dropout=0.9,
    dropout_decay=1e-3,
    nb_epoch=20,
    L2=1e-6,
    device="cpu",
):
    cfg = dict(locals())
    print(cfg, file=sys.stderr)
    if cupy is not None and device != "cpu":
        print("Using GPU", file=sys.stderr)
        Model.ops = CupyOps()
        Model.ops.device = device
    train_data, check_data, tag_map = twitter_ner()
    dev_words, dev_tags = zip(*check_data)
    nr_tag = len(tag_map)

    extracter = FeatureExtracter("en", attrs=[ORTH, LOWER, SHAPE, PREFIX, SUFFIX])
    Model.lsuv = True
    with Model.define_operators({"**": clone, ">>": chain, "+": add, "|": concatenate}):
        glove = StaticVectors("en", width // 2, column=0)
        lower_case = HashEmbed(width, 500, column=1) + HashEmbed(width, 100, column=1)
        shape = HashEmbed(width // 2, 200, column=2)
        prefix = HashEmbed(width // 2, 100, column=3)
        suffix = HashEmbed(width // 2, 100, column=4)

        model = (
            layerize(flatten_sequences)
            >> (lower_case | shape | prefix | suffix)
            >> BN(Maxout(width, pieces=3), nO=width)
            >> Residual(ExtractWindow(nW=1) >> BN(Maxout(width, pieces=3), nO=width))
            ** depth
            >> Softmax(nr_tag)
        )

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

            backprop(yh - y, optimizer)

            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001

            epoch_train_acc += (yh.argmax(axis=1) == y.argmax(axis=1)).sum()
            # if epoch_train_acc / n_train >= 0.999:
            #    break
    with model.use_params(trainer.optimizer.averages):
        print(model.evaluate(dev_X, model.ops.flatten(dev_y)), file=sys.stderr)
        print_dev_sentences(model, dev_words, dev_tags, dev_X, tag_map)


if __name__ == "__main__":
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats

        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
