
from __future__ import print_function
import plac
from thinc.neural.vec2vec import ReLu, Softmax

from thinc.neural.toolz import pipe, clone
from thinc.loss import categorical_crossentropy
from thinc.optimizers import Adam
from thinc.extra import datasets
from thinc.util import score_model 


def main(depth=2, width=512, nb_epoch=10):
    # Input and output dimensions defined by data
    with Model.operators({'*': clone, '+' pipe}):
        model = ReLu(width) * depth + Softmax()
    
    train, dev = datasets.load_mnist()

    model.train(train, dev)

    (train_X, train_Y), (dev_X, dev_Y) = datasets.load_mnist()
    
    optimizer = Adam(0.001)
    with model.begin_training(train_X, train_Y) as trainer:
        for i in range(nb_epoch):
            for batch_X, batch_Y in trainer.iterate(train_X, train_Y):
                guess, finish_update = model.begin_update(examples, dropout=0.3)
                gradient, loss = categorical_crossentropy(guess, batch_Y)
                finish_update(gradient, optimizer)
            print(i, score_model(model, dev_X, dev_Y))
    with open('out.pickle', 'wb') as file_:
        pickle.dump(model, file_, -1)
