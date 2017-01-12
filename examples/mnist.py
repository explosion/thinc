from __future__ import print_function
import plac
from thinc.neural.vec2vec import Model, Affine, ReLu, ELU, Softmax
from thinc.neural.vec2vec import Residual

from thinc.loss import categorical_crossentropy
from thinc.extra import datasets
from thinc.neural.util import score_model 
from thinc.neural.ops import NumpyOps
from cytoolz import curry
from thinc.api import layerize

try:
    import cPickle as pickle
except ImportError:
    import pickle


@curry
def health_check(name, X, **kwargs):
    print(name, X.mean(), X.var())
    return X, lambda grad, *args, **kwargs: grad


def main(depth=2, width=512, nb_epoch=20):
    def Block(width):
        with Model.bind_operators({'>>': chain}):
            block = batch_norm >> Rescale() >> relu >> Affine(width)
        return block 

    with Model.bind_operators({'*': clone, '>>': chain, '//': residual}):
        model = (
            Affine(width, 784)
            >> depth * (Block(width) // Block(width))
            >> Softmax(10)
        )
    
    (train_X, train_Y), (dev_X, dev_Y), (test_X, test_Y) = datasets.mnist()

    with model.begin_training(train_X, train_Y) as (trainer, optimizer):
        while trainer.next_epoch:
            for examples, truths in trainer.iterate(train_X, train_Y):
                with model.set_dropout(trainer.dropout):
                    guesses, backprop = model.begin_update(X)

                loss, gradient_of_loss = categorical_crossentropy(guesses, truth)

                backprop(gradient_of_loss)
                
                for name, param, grad in model.pending_updates():
                    optimizer(param, grad, name=name, loss=loss)
                
                trainer.record_train_loss(loss)
            with model.use_params(optimizer.averages):
                dev_acc_avg = model.evaluate(dev_X, dev_Y)
                print('Avg dev.: %.3f' % dev_acc_avg)
            trainer.record_dev_acc(dev_acc_avg)

    with model.use_params(optimizer.averages):
        print('Avg dev.: %.3f' % score_model(model, dev_X, dev_Y))
        print('Avg test.: %.3f' % score_model(model, test_X, test_Y))
        with open('out.pickle', 'wb') as file_:
            pickle.dump(model, file_, -1)


if __name__ == '__main__':
    plac.call(main)
