from __future__ import print_function
from thinc.linear.linear import LinearModel
from thinc.neural._classes.model import Model
from thinc.extra import datasets
from thinc.neural.util import to_categorical
import spacy
from spacy.attrs import ORTH
from timeit import default_timer as timer


def preprocess(ops, keys):
    lengths = ops.asarray([arr.shape[0] for arr in keys])
    keys = ops.xp.concatenate(keys)
    vals = ops.allocate(keys.shape[0]) + 1
    return keys, vals, lengths


epoch_train_acc = 0.
def track_progress(**context):
    model = context['model']
    dev_X = context['dev_X']
    dev_y = model.ops.flatten(context['dev_y'])
    n_train = context['n_train']
    trainer = context['trainer']
    n_dev = len(dev_y)
    epoch_times = [timer()]
    def each_epoch():
        global epoch_train_acc
        epoch_start = epoch_times[-1]
        epoch_end = timer()
        wps_train = n_train / (epoch_end-epoch_start)
        dev_start = timer()
        acc = model.evaluate(dev_X, dev_y)
        dev_end = timer()
        wps_run = n_dev / (dev_end-dev_start)
        with model.use_params(trainer.optimizer.averages):
            avg_acc = model.evaluate(dev_X, dev_y)
        stats = (acc, avg_acc, float(epoch_train_acc) / n_train, trainer.dropout,
                 wps_train, wps_run)
        print("%.3f (%.3f) dev acc, %.3f train acc, %.4f drop, %d wps train, %d wps run" % stats)
        epoch_train_acc = 0.
        epoch_times.append(timer())
    return each_epoch


def main():
    train, dev = datasets.imdb()
    train_X, train_y = zip(*train)
    dev_X, dev_y = zip(*dev)
    model = LinearModel(2)
    train_y = to_categorical(train_y, nb_classes=2)
    dev_y = to_categorical(dev_y, nb_classes=2)

    nlp = spacy.load('en')
    train_X = [model.ops.asarray([tok.orth for tok in doc if not tok.is_stop], dtype='uint64')
               for doc in nlp.pipe(train_X)]
    dev_X = [model.ops.asarray([tok.orth for tok in doc if not tok.is_stop], dtype='uint64')
               for doc in nlp.pipe(dev_X)]
    dev_X = preprocess(model.ops, dev_X)
    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        n_train = len(train_X)
        trainer.each_epoch.append(track_progress(**locals()))
        trainer.batch_size = 64
        trainer.nb_epoch = 3
        for X, y in trainer.iterate(train_X, train_y):
            keys_vals_lens = preprocess(model.ops, X)
            scores, backprop = model.begin_update(keys_vals_lens)
            backprop(scores-y, optimizer)



if __name__ == '__main__':
    main()
