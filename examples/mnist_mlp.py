import plac
from srsly import cloudpickle as pickle
from thinc.v2v import Model, ReLu, Softmax
from thinc.api import clone, chain
from thinc.neural.util import to_categorical, prefer_gpu
from thinc.extra import datasets


def main(depth=2, width=512, nb_epoch=30):
    prefer_gpu()
    # Configuration here isn't especially good. But, for demo..
    with Model.define_operators({"**": clone, ">>": chain}):
        model = ReLu(width) >> ReLu(width) >> Softmax()

    train_data, dev_data, _ = datasets.mnist()
    train_X, train_y = model.ops.unzip(train_data)
    dev_X, dev_y = model.ops.unzip(dev_data)

    dev_y = to_categorical(dev_y)
    with model.begin_training(train_X, train_y, L2=1e-6) as (trainer, optimizer):
        epoch_loss = [0.0]

        def report_progress():
            with model.use_params(optimizer.averages):
                print(epoch_loss[-1], model.evaluate(dev_X, dev_y), trainer.dropout)
            epoch_loss.append(0.0)

        trainer.each_epoch.append(report_progress)
        trainer.nb_epoch = nb_epoch
        trainer.dropout = 0.3
        trainer.batch_size = 128
        trainer.dropout_decay = 0.0
        train_X = model.ops.asarray(train_X, dtype="float32")
        y_onehot = to_categorical(train_y)
        for X, y in trainer.iterate(train_X, y_onehot):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            loss = ((yh - y) ** 2.0).sum() / y.shape[0]
            backprop(yh - y, optimizer)
            epoch_loss[-1] += loss
        with model.use_params(optimizer.averages):
            print("Avg dev.: %.3f" % model.evaluate(dev_X, dev_y))
            with open("out.pickle", "wb") as file_:
                pickle.dump(model, file_, -1)


if __name__ == "__main__":
    plac.call(main)
