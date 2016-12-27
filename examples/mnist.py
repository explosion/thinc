from __future__ import print_function
import plac

import tqdm
import numpy as np
from thinc import datasets
from thinc.base import Network
from thinc.vec2vec import Affine, ReLu, Softmax
from thinc.util import score_model
from thinc.optimizers import linear_decay


class ReLuMLP(Network):
    Hidden = ReLu
    Output = Softmax
    width = 128
    depth = 3

    def setup(self, nr_out, nr_in, **kwargs):
        for i in range(self.depth):
            self.layers.append(self.Hidden(nr_out=self.width, nr_in=nr_in,
                name='hidden-%d' % i))
            nr_in = self.width
        self.layers.append(self.Output(nr_out=nr_out, nr_in=nr_in))
        self.set_weights()
        self.set_gradient()


def get_gradient(scores, labels):
    target = np.zeros(scores.shape)
    for i, label in enumerate(labels):
        target[i, int(label)] = 1.0
    return scores - target


def main(batch_size=128, nb_epoch=10, nb_classes=10):
    model = ReLuMLP(10, 784)
    print([(layer.nr_out, layer.nr_in) for layer in model.layers])
    train_data, check_data, test_data = datasets.keras_mnist()
    
    with model.begin_training(train_data) as (trainer, optimizer):
        print(len(train_data))
        for examples, truth in trainer.iterate(model, train_data, check_data,
                                               nb_epoch=nb_epoch):
            assert hasattr(examples, 'shape'), type(examples)
            guess, finish_update = model.begin_update(examples, dropout=0.3)
            gradient, loss = trainer.get_gradient(guess, truth)
            optimizer.set_loss(loss)
            finish_update(gradient, optimizer)
        #print(loss / len(truth))
    print('Test score:', score_model(model, test_data))


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
