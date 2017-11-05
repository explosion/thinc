import plac
import numpy

from torch import autograd
from torch import nn
import torch.optim

from thinc.extra.wrappers import PytorchWrapper


def main(length=1000, nO=32, nI=32):
    pt_model = nn.Linear(nI, nO)
    optimizer = torch.optim.Adam(pt_model.parameters())

    model = PytorchWrapper(pt_model)

    X = numpy.ones((length, nI), dtype='f')
    y = 1. / X
    for i in range(10):
        yh, get_dX = model.begin_update(X)
        dY = (yh - y) / len(y)
        dX = get_dX(dY)
        optimizer.step()


if __name__ == '__main__':
    plac.call(main)
