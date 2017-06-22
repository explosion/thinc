from __future__ import print_function
from thinc.neural._classes.resnet import Residual
from thinc.neural._classes.convolution import ExtractWindow
from thinc.neural import Model, ReLu, Maxout, Softmax, Affine
from thinc.neural._classes.selu import SELU
from thinc.neural._classes.batchnorm import BatchNorm as BN

from thinc.neural._classes.attention import ParametricAttention

from thinc.neural.pooling import Pooling, sum_pool, max_pool, mean_pool
from thinc.extra import datasets
from thinc.neural.util import to_categorical
from thinc.neural._classes.hash_embed import HashEmbed
from thinc.api import chain, concatenate, clone
from thinc.api import foreach_sentence, uniqued
from thinc.api import layerize, with_flatten, flatten_add_lengths, with_getitem
from thinc.api import FeatureExtracter
import spacy
from spacy.attrs import ORTH, LOWER, SHAPE, PREFIX, SUFFIX


def build_model(nr_class, width):
    with Model.define_operators({'|': concatenate, '>>': chain, '**': clone}):
        embed = (
            (HashEmbed(width, 5000, column=1)
            | HashEmbed(width, 750, column=2)
            | HashEmbed(width, 750, column=3)
            | HashEmbed(width, 5000, column=4))
            >> Maxout(width)
        )

        sent2vec = (
            FeatureExtracter([ORTH, LOWER, PREFIX, SUFFIX, SHAPE])
            >> flatten_add_lengths
            >> with_getitem(0,
                uniqued(embed, column=0)
                >> Residual(ExtractWindow(nW=1) >> Maxout(width)) ** 2
            )
            >> ParametricAttention(width)
            >> Pooling(max_pool)
            >> Residual(Maxout(width)) ** 2
        )

        model = (
            foreach_sentence(sent2vec, drop_factor=4.0)
            >> flatten_add_lengths
            >> ParametricAttention(width)
            >> Pooling(sum_pool)
            >> Residual(Maxout(width)) ** 2
            >> Softmax(nr_class)
        )
    model.lsuv = False
    return model


def main():
    train, dev = datasets.imdb()
    train_X, train_y = zip(*train)
    dev_X, dev_y = zip(*dev)
    model = build_model(2, 128)
    train_y = to_categorical(train_y, nb_classes=2)
    dev_y = to_categorical(dev_y, nb_classes=2)

    nlp = spacy.load('en')
    nlp.vocab.lex_attr_getters[PREFIX] = lambda string: string[:3]
    for word in nlp.vocab:
        word.prefix_ = word.orth_[:3]

    print("Create data")
    #train_X = train_X[:1000]
    #train_y = train_y[:1000]
    train_X = list(nlp.pipe(train_X))
    dev_X = list(nlp.pipe(dev_X))
    dev_X = dev_X[:1000]
    dev_y = dev_y[:1000]
    print("Train")
    with model.begin_training(train_X[:100], train_y[:100], L2=1e-6) as (trainer, optimizer):
        trainer.dropout = 0.2
        trainer.batch_size = 128
        trainer.nb_epoch = 30
        def report_progress():
            with model.use_params(optimizer.averages):
                print(loss, model.evaluate(dev_X, dev_y))
        trainer.each_epoch.append(report_progress)
        loss = 0.
        for Xs, ys in trainer.iterate(train_X, train_y):
            yhs, backprop = model.begin_update(Xs, drop=trainer.dropout)
            backprop((yhs-ys)/ys.shape[0], optimizer)
            loss += ((yhs-ys)**2).sum()
    with model.use_params(optimizer.averages):
        print(model.evaluate(dev_X, dev_y))
 

if __name__ == '__main__':
    main()
