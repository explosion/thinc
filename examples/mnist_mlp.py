from __future__ import print_function
import plac

from thinc.neural.vec2vec import ReLu, Softmax
from thinc.api import clone, chain

from thinc.extra import datasets
from thinc.neural.loss import categorical_crossentropy
from thinc.neural.util import score_model 

import pickle


def main(depth=2, width=512, nb_epoch=20):
    with Model.define_operators({'*': clone, '>>': chain}):
        model = ReLu(width) * depth >> Softmax()
    
    (train_X, train_Y), (dev_X, dev_Y), (test_X, test_Y) = datasets.mnist()

    with model.begin_training(train_X, train_Y):
        optimizer = Adam(0.001)
        for i in range(nb_epoch):
            for X, y in (train_X, train_Y):
                yh, backprop = model.begin_update(X)
                loss, d_loss = categorical_crossentropy(y, yh)
                backprop(d_loss)
                for name, param, d_param in model.weights:
                    if d_param is not None:
                        optimizer(param, d_param, key=name)
                
            with model.use_params(optimizer.averages):
                dev_acc_avg = model.evaluate(dev_X, dev_Y)
                print('Avg dev.: %.3f' % dev_acc_avg)

    with model.use_params(optimizer.averages):
        print('Avg dev.: %.3f' % model.evaluate(dev_X, dev_Y))
        print('Avg test.: %.3f' % model.evaluate(test_X, test_Y))
        with open('out.pickle', 'wb') as file_:
            pickle.dump(model, file_, -1)


if __name__ == '__main__':
    plac.call(main)
