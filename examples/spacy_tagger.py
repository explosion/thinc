# coding: utf8
from __future__ import print_function, unicode_literals, division

from timeit import default_timer as timer
import numpy
import spacy
from spacy.tokens import Doc
import numpy.random
import numpy.linalg
import plac

from thinc.extra import datasets
from thinc.neural.vec2vec import Model, Maxout
from thinc.neural.vec2vec import Softmax, Affine
from thinc.neural._classes.batchnorm import BatchNorm
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural.util import to_categorical
from thinc.api import layerize, chain, concatenate, clone, with_flatten
from spacy._ml import SpacyVectors


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
                _ = nlp.vocab[word]  # noqa: F841
            X.append(Doc(nlp.vocab, words=words))
            y.append([tagmap[tag] for tag in tags])
            oovs += sum(not w.has_vector for w in X[-1])
            n += len(X[-1])
        print(oovs, n, oovs / (n+1e-10))
        return zip(X, y)

    return _encode(train_sents), _encode(dev_sents), len(tagmap)


@layerize
def get_positions(ids, drop=0.0):
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
    print("Loading spaCy model")
    nlp = spacy.load("en_core_web_md", pipeline=[])
    train_sents, dev_sents, _ = datasets.ewtb_pos_tags()
    train_sents, dev_sents, nr_class = spacy_preprocess(nlp, train_sents, dev_sents)
    if nr_sent >= 1:
        train_sents = train_sents[:nr_sent]

    print("Building the model")
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        model = (
            SpacyVectors
            >> with_flatten(
              Affine(width)
              >> (ExtractWindow(nW=1) >> BatchNorm(Maxout(width))) ** depth
              >> Softmax(nr_class)
          )
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
        epoch_loss = [0.0]
        n_train = sum(len(y) for y in train_y)

        def track_progress():
            start = timer()
            acc = model.evaluate(dev_X, dev_y)
            end = timer()
            with model.use_params(optimizer.averages):
                avg_acc = model.evaluate(dev_X, dev_y)
            stats = (
                epoch_loss[-1],
                acc,
                avg_acc,
                n_train,
                (end - epoch_times[-1]),
                n_train / (end - epoch_times[-1]),
                len(dev_y),
                (end - start),
                float(dev_y.shape[0]) / (end - start),
                trainer.dropout,
            )
            print(
                len(epoch_loss),
                "%.3f train, %.3f (%.3f) dev, %d/%d=%d wps train, %d/%.3f=%d wps run. d.o.=%.3f"
                % stats,
            )
            epoch_times.append(end)
            epoch_loss.append(0.0)

        trainer.each_epoch.append(track_progress)
        print("Training")
        batch_size = 1.0
        for examples, truth in trainer.iterate(train_X, train_y):
            truth = to_categorical(model.ops.flatten(truth), nb_classes=50)
            guess, finish_update = model.begin_update(examples, drop=trainer.dropout)
            n_correct = (guess.argmax(axis=1) == truth.argmax(axis=1)).sum()
            finish_update(guess - truth, optimizer)
            epoch_loss[-1] += n_correct / n_train
            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001
    with model.use_params(optimizer.averages):
        print("End: %.3f" % model.evaluate(dev_X, dev_y))


if __name__ == "__main__":
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats

        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(20)
