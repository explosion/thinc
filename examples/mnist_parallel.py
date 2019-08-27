# coding: utf8
from __future__ import unicode_literals, print_function

import ray
import plac
from srsly import cloudpickle as pickle
from thinc.v2v import Model, ReLu, Softmax
from thinc.api import clone, chain
from thinc.neural.util import to_categorical, prefer_gpu
from thinc.extra.param_server import ParamServer, parallel_train

from thinc.extra import datasets


def main(depth=2, width=512, nb_epoch=6, batch_size=128, nproc=4):
    prefer_gpu()
    ray.init(object_store_memory=3000000000, num_cpus=nproc+1)
    # Configuration here isn't especially good. But, for demo..
    with Model.define_operators({"**": clone, ">>": chain}):
        model = ReLu(width) >> ReLu(width) >> Softmax()

    train_data, dev_data, _ = datasets.mnist()
    train_X, train_y = model.ops.unzip(train_data)
    dev_X, dev_y = model.ops.unzip(dev_data)

    dev_y = to_categorical(dev_y)
    train_y = to_categorical(train_y)
    with model.begin_training(train_X, train_y, L2=1e-6) as (_, optimizer):
        param_server = ParamServer(model)

        for i in range(nb_epoch):
            loss = parallel_train(param_server, model, optimizer, train_X, train_y,
                batch_size, nproc)
            print(i, loss, model.evaluate(dev_X, dev_y))


if __name__ == "__main__":
    plac.call(main)
