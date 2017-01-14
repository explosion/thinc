from __future__ import print_function
import plac
import dill as pickle

from thinc.neural.vec2vec import Model, ReLu, Softmax
from thinc.neural._classes.batchnorm import BatchNorm as BN
from thinc.api import clone, chain
from thinc.loss import categorical_crossentropy

from thinc.extra import datasets


def main(depth=4, width=512, nb_epoch=5):
    with Model.define_operators({'**': clone, '>>': chain}):
        model = BN(ReLu(width, 784)) \
                >> BN(ReLu(width)) \
                >> Softmax()
   
    train_data, dev_data, _ = datasets.mnist()
    train_X, train_y = model.ops.unzip(train_data)
    dev_X, dev_y = model.ops.unzip(dev_data)

    with model.begin_training(train_X, train_y) as (trainer, optimizer):
        trainer.each_epoch(lambda: print(model.evaluate(dev_X, dev_y)))
        trainer.nb_epoch = nb_epoch
        trainer.dropout = 0.2
        trainer.dropout_decay = 0.0
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            d_loss, loss = categorical_crossentropy(yh, y)
            optimizer.set_loss(loss)
            backprop(d_loss, optimizer)
    with model.use_params(optimizer.averages):
        print('Avg dev.: %.3f' % model.evaluate(dev_X, dev_y))
        with open('out.pickle', 'wb') as file_:
            pickle.dump(model, file_, -1)


if __name__ == '__main__':
    plac.call(main)
