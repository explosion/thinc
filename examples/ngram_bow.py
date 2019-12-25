from srsly import cloudpickle as pickle
from thinc.v2v import Model, Softmax
from thinc.t2v import Pooling, mean_pool
from thinc.extra import datasets
from thinc.neural.util import to_categorical
from thinc.neural._classes.hash_embed import HashEmbed
from thinc.api import chain, concatenate, clone, uniqued
from thinc.api import flatten_add_lengths, with_getitem
from thinc.api import FeatureExtracter
from thinc.neural.ops import CupyOps
from spacy.language import Language
from spacy.attrs import ORTH


def build_model(nr_class, width, **kwargs):
    with Model.define_operators({"|": concatenate, ">>": chain, "**": clone}):
        model = (
            FeatureExtracter([ORTH])
            >> flatten_add_lengths
            >> with_getitem(0, uniqued(HashEmbed(width, 10000, column=0)))
            >> Pooling(mean_pool)
            >> Softmax(nr_class)
        )
    model.lsuv = False
    return model


def main(use_gpu=False, nb_epoch=50):
    if use_gpu:
        Model.ops = CupyOps()
        Model.Ops = CupyOps
    train, test = datasets.imdb()
    print("Load data")
    train_X, train_y = zip(*train)
    test_X, test_y = zip(*test)
    train_y = to_categorical(train_y, nb_classes=2)
    test_y = to_categorical(test_y, nb_classes=2)

    nlp = Language()

    dev_X = train_X[-1000:]
    dev_y = train_y[-1000:]
    train_X = train_X[:-1000]
    train_y = train_y[:-1000]
    print("Parse data")
    train_X = [nlp.make_doc(x) for x in train_X]
    dev_X = [nlp.make_doc(x) for x in dev_X]

    model = build_model(2, 1)

    print("Begin training")
    with model.begin_training(train_X, train_y, L2=1e-6) as (trainer, optimizer):
        epoch_loss = [0.0]

        def report_progress():
            with model.use_params(optimizer.averages):
                print(epoch_loss[-1], model.evaluate(dev_X, dev_y), trainer.dropout)
            epoch_loss.append(0.0)

        trainer.each_epoch.append(report_progress)
        trainer.nb_epoch = nb_epoch
        trainer.dropout = 0.0
        trainer.batch_size = 128
        trainer.dropout_decay = 0.0
        for X, y in trainer.iterate(train_X[:1000], train_y[:1000]):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            loss = ((yh - y) ** 2.0).sum() / y.shape[0]
            backprop((yh - y) / y.shape[0], optimizer)
            epoch_loss[-1] += loss
        with model.use_params(optimizer.averages):
            print("Avg dev.: %.3f" % model.evaluate(dev_X, dev_y))
            with open("out.pickle", "wb") as file_:
                pickle.dump(model, file_, -1)


if __name__ == "__main__":
    main()
