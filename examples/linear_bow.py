from __future__ import print_function
from thinc.linear.linear import LinearModel
from thinc.neural._classes.model import Model
from thinc.extra import datasets
from thinc.neural.util import to_categorical
import spacy
from spacy.attrs import ORTH


def preprocess(ops, keys):
    lengths = ops.asarray([arr.shape[0] for arr in keys])
    keys = ops.xp.concatenate(keys)
    vals = ops.allocate(keys.shape[0]) + 1
    return keys, vals, lengths


def main():
    train, dev = datasets.imdb()
    train_X, train_y = zip(*train)
    dev_X, dev_y = zip(*dev)
    model = LinearModel(2)
    train_y = to_categorical(train_y, nb_classes=2)
    dev_y = to_categorical(dev_y, nb_classes=2)

    nlp = spacy.load('en')
    train_X = [model.ops.asarray([tok.orth for tok in doc], dtype='uint64')
               for doc in nlp.pipe(train_X)]
    dev_X = [model.ops.asarray([tok.orth for tok in doc], dtype='uint64')
               for doc in nlp.pipe(dev_X)]
    dev_X = preprocess(model.ops, dev_X)
    with model.begin_training(train_X, train_y, L2=1e-6) as (trainer, optimizer):
        trainer.dropout = 0.0
        trainer.batch_size = 512
        trainer.nb_epoch = 3
        trainer.each_epoch.append(lambda: print(model.evaluate(dev_X, dev_y)))
        for X, y in trainer.iterate(train_X, train_y):
            keys_vals_lens = preprocess(model.ops, X)
            scores, backprop = model.begin_update(keys_vals_lens,
                    drop=trainer.dropout)
            backprop(scores-y, optimizer)
    with model.use_params(optimizer.averages):
        print(model.evaluate(dev_X, dev_y))
 


if __name__ == '__main__':
    main()
