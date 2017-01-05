from __future__ import print_function
import plac
from thinc.neural.vec2vec import Model, ReLu, Softmax

from thinc.loss import categorical_crossentropy
from thinc.neural.optimizers import Adam
from thinc.extra import datasets
from thinc.neural.util import score_model 
from thinc.neural.ops import NumpyOps

try:
    import cPickle as pickle
except ImportError:
    import pickle


def main(depth=2, width=512, nb_epoch=10):
    model = Model(
              ReLu(128, 784, name='relu1'),
              ReLu(128, 128, name='relu2'),
              ReLu(128, 128, name='relu3'),
              Softmax(10, 128, name='softmax'),
              ops=NumpyOps())
    
    train_data, dev_data, test_data = datasets.mnist()
    train_X, train_Y = zip(*train_data)
    dev_X, dev_Y = zip(*dev_data)
    test_X, test_Y = zip(*test_data)
    dev_X = model.ops.asarray(dev_X)
    dev_Y = model.ops.asarray(dev_Y)
    test_X = model.ops.asarray(test_X)
    test_Y = model.ops.asarray(test_Y)

    with model.begin_training(train_data) as (trainer, optimizer):
        for i in range(nb_epoch):
            for batch_X, batch_Y in trainer.iterate(
                    model, train_data, dev_X, dev_Y, nb_epoch=1):
                batch_X = model.ops.asarray(batch_X)
                guess, finish_update = model.begin_update(batch_X,
                                         dropout=trainer.dropout)
                gradient, loss = categorical_crossentropy(guess, batch_Y)
                optimizer.set_loss(loss)
                trainer._loss += loss / len(batch_Y)
                finish_update(gradient, optimizer)
            with model.use_params(optimizer.averages):
                print('Avg dev.: %.3f' % score_model(model, dev_X, dev_Y))
    with model.use_params(optimizer.averages):
        print('Avg dev.: %.3f' % score_model(model, dev_X, dev_Y))
        print('Avg test.: %.3f' % score_model(model, test_X, test_Y))
        with open('out.pickle', 'wb') as file_:
            pickle.dump(model, file_, -1)


if __name__ == '__main__':
    plac.call(main)
