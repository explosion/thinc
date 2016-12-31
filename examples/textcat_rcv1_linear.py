from __future__ import print_function
import plac

import tqdm
import numpy as np
import random
from thinc.extra import datasets
from thinc.linear.avgtron import AveragedPerceptron
from thinc.linear.features import BagOfWords


def score_model(model, X, y):
    acc = 0.
    for i in range(len(X)):
        scores = model(X[i])
        acc += scores.argmax() == y[i]
    return acc / len(X)


def main(nb_epoch=10):
    (X_train, y_train), (X_test, y_test) = datasets.reuters()
    model = AveragedPerceptron(extracter=BagOfWords(), nr_out=max(y_train)+1)
    X_train = [np.asarray(sorted(x), dtype='uint64') for x in X_train]
    y_train = np.asarray(y_train, dtype='uint64')
    X_test = [np.asarray(sorted(x), dtype='uint64') for x in X_test]
    y_test = np.asarray(y_test, dtype='uint64')
    
    train_data = (X_train, y_train)
    check_data = (X_test, y_test)

    with model.begin_training(train_data) as (trainer, _):
        for examples, truth in trainer.iterate(model, train_data, check_data,
                                               nb_epoch=nb_epoch):
            guess, finish_update = model.begin_update(examples)
            gradient, loss = trainer.get_gradient(guess, truth)
            #optimizer.set_loss(loss)
            finish_update(truth)
    print('Dev: %.3f' % score_model(model, X_test, y_test))


if __name__ == '__main__':
    plac.call(main)

