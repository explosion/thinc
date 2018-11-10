import plac
import os
import cProfile
import pstats
from thinc.misc import LayerNorm
from thinc.v2v import Model
import random
import time
from thinc.api import layerize, noop
random.seed(0)

class DummyChild(Model):
    def __init__(self, nO):
        self.nO = nO
        Model.__init__(self)

    def begin_update(self, X, drop=0.):
        return X, None


def create_data(ops, nr_col, n_samples=50):
    for i in range(n_samples):
        nr_row = int(100 * random.random())
        yield ops.xavier_uniform_init(ops.allocate((nr_row, nr_col)))


def main(nr_col=128):
    Model().ops.xp.random.seed(0)
    model = LayerNorm(DummyChild(nr_col))
    Xs = list(create_data(model.ops, nr_col))
    with model.begin_training(Xs[0]):
        pass
    total_Y = 0.
    start = time.time()
    for i in range(1):
        for X in Xs:
            Y, get_dX = model.begin_update(X)
            print(i, Y.sum())
            total_Y += Y.sum()
    end = time.time()
    print(end-start, total_Y)


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(5)

        
