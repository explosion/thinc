from __future__ import print_function, division
import plac
import numpy

from thinc.neural.id2vec import Embed
from thinc.neural.vec2vec import Model, Maxout, ReLu, Softmax
from thinc.neural._classes.convolution import ExtractWindow

from thinc.api import layerize, chain, clone
from thinc.neural.util import flatten_sequences, remap_ids, to_categorical
from thinc.neural.ops import NumpyOps, CupyOps
from thinc.neural.optimizers import SGD

from thinc.extra.datasets import ancora_pos_tags


epoch_train_acc = 0.
def track_progress(**context):
    model = context['model']
    dev_X = context['dev_X']
    dev_y = model.ops.flatten(context['dev_y'])
    n_train = context['n_train']
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


remapping = layerize(remap_ids(NumpyOps()))
def preprocess(ops, data, nr_tag):
    Xs, ys = zip(*data)
    Xs = [ops.asarray(remapping(x)) for x in Xs]
    ys = [ops.asarray(to_categorical(y, nb_classes=nr_tag)) for y in ys]
    return Xs, ys

_i = 0
def debug(X, drop=0.):
    global _i
    if _i % 1000 == 0:
        print(X.mean(), X.var())
    _i += 1
    return X, lambda d, sgd: d


def main(width=128, depth=4, vector_length=64, max_batch_size=32,
        dropout=0.9, drop_decay=1e-4, nb_epoch=20, L2=1e-5):
    cfg = dict(locals())
    Model.ops = CupyOps()
    train_data, check_data, nr_tag = ancora_pos_tags()
    
    with Model.define_operators({'**': clone, '>>': chain}):
        model = (
            layerize(flatten_sequences)
            >> Embed(width, vector_length)
            >> (ExtractWindow(nW=1) >> Maxout(width, pieces=3)) ** depth
            >> Softmax(nr_tag))

    train_X, train_y = preprocess(model.ops, train_data, nr_tag)
    dev_X, dev_y = preprocess(model.ops, check_data, nr_tag)

    n_train = float(sum(len(x) for x in train_X))
    global epoch_train_acc
    with model.begin_training(train_X, train_y, **cfg) as (trainer, optimizer):
        trainer.each_epoch.append(track_progress(**locals()))
        trainer.batch_size = 1
        batch_size = 1.
        for X, y in trainer.iterate(train_X, train_y):
            y = model.ops.flatten(y)
            
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            backprop(yh - y, optimizer)
            
            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001
            
            epoch_train_acc += (yh.argmax(axis=1) == y.argmax(axis=1)).sum()
    with model.use_params(trainer.optimizer.averages):
        print(model.evaluate(dev_X, model.ops.flatten(dev_y)))
 

if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
