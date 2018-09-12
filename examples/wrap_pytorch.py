import plac
import numpy

import torch
from torch import autograd
from torch import nn
import torch.optim
import torch.cuda
from thinc.neural.ops import CupyOps

from thinc.extra.wrappers import PyTorchWrapper
from thinc.v2v import Model


def main(length=1000, nO=32, nI=32):
    if CupyOps.xp != None:
        print("Use GPU")
        Model.ops = CupyOps()
        Model.Ops = CupyOps
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    pt_model = nn.Linear(nI, nO)
    optimizer = torch.optim.Adam(pt_model.parameters())

    model = PyTorchWrapper(pt_model)

    X = Model.ops.xp.ones((length, nI), dtype='f')
    y = 1. / X
    for i in range(10):
        yh, get_dX = model.begin_update(X)
        dY = (yh - y) / len(y)
        dX = get_dX(dY)


if __name__ == '__main__':
    plac.call(main)
