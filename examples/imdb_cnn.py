from __future__ import print_function
import tqdm

from thinc.v2v import Model, SELU, ReLu, Maxout, Softmax, Affine
from thinc.t2t import ExtractWindow
from thinc.misc import BatchNorm as BN
from thinc.misc import Residual

from thinc.t2t import ParametricAttention

from thinc.neural.pooling import Pooling, sum_pool, max_pool, mean_pool
from thinc.extra import datasets
from thinc.neural.util import to_categorical
from thinc.neural._classes.hash_embed import HashEmbed
from thinc.api import layerize, chain, concatenate, clone
from thinc.api import foreach, foreach_sentence, uniqued
from thinc.api import layerize, with_flatten, flatten_add_lengths, with_getitem
from thinc.api import FeatureExtracter
import spacy
from spacy.attrs import ORTH, LOWER, SHAPE, PREFIX, SUFFIX

from thinc.extra.hpbff import BestFirstFinder, train_epoch
from thinc.neural.ops import CupyOps


@layerize
def get_sents(docs, drop=0.):
    sents = [list(doc.sents) for doc in docs]
    return sents, None


def build_model(nr_class, width, depth, conv_depth, **kwargs):
    with Model.define_operators({'|': concatenate, '>>': chain, '**': clone}):
        embed = (
            (HashEmbed(width, 5000, column=1)
            | HashEmbed(width//2, 750, column=2)
            | HashEmbed(width//2, 750, column=3)
            | HashEmbed(width//2, 750, column=4))
            >> Maxout(width)
        )

        sent2vec = (
            flatten_add_lengths
            >> with_getitem(0,
                uniqued(embed, column=0)
                >> Residual(ExtractWindow(nW=1) >> SELU(width)) ** conv_depth
            )
            >> ParametricAttention(width)
            >> Pooling(sum_pool)
            >> Residual(SELU(width)) ** depth
        )

        model = (
            foreach(sent2vec, drop_factor=2.0)
            >> flatten_add_lengths
            >> ParametricAttention(width, hard=False)
            >> Pooling(sum_pool)
            >> Residual(SELU(width)) ** depth
            >> Softmax(nr_class)
        )
    model.lsuv = False
    return model


def simple_train(nlp, model_data, train_X, train_y, dev_X, dev_y):
    model, sgd, hp = model_data
    for i in range(10):
        _, ((model, sgd, hp), train_acc, dev_acc) = train_epoch(model, sgd, hp,
                                                        train_X, train_y,
                                                        dev_X, dev_y,
                                                        device_id=0)
        print(train_acc, dev_acc)
    with model.use_params(optimizer.averages):
        dev_acc = model.evaluate(dev_X, dev_y)
    return (model, optimizer, hp), train_acc, dev_acc


def main(use_gpu=True):
    if use_gpu:
        Model.ops = CupyOps()
        Model.Ops = CupyOps
    train, test = datasets.imdb()
    print("Load data")
    train_X, train_y = zip(*train)
    test_X, test_y = zip(*test)
    train_y = Model.ops.asarray(to_categorical(train_y, nb_classes=2))
    test_y = Model.ops.asarray(to_categorical(test_y, nb_classes=2))
    
    #train_X = train_X[:2000]
    #test_X = test_X[:2000]

    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    nlp.vocab.lex_attr_getters[PREFIX] = lambda string: string[:3]
    for word in nlp.vocab:
        word.prefix_ = word.orth_[:3]

    preprocessor = FeatureExtracter([ORTH, LOWER, PREFIX, SUFFIX, SHAPE])
    train_X = [preprocessor(list(doc.sents)) for doc in tqdm.tqdm(nlp.pipe(train_X))]
    test_X = [preprocessor(list(doc.sents)) for doc in tqdm.tqdm(nlp.pipe(test_X))]

    dev_X = train_X[-1000:]
    dev_y = train_y[-1000:]
    train_X = train_X[:-1000]
    train_y = train_y[:-1000]
    print("Parse data")
    n_sent = sum([len(list(sents)) for sents in train_X])
    print("%d sentences" % n_sent)

    hpsearch = BestFirstFinder(
                 nonlin=[SELU],
                 width=[64],
                 depth=[2],
                 conv_depth=[2],
                 batch_size=[128],
                 learn_rate=[0.001],
                 L2=[1e-6],
                 beta1=[0.9],
                 beta2=[0.999],
                 dropout=[0.2],
                 nr_update=[len(train_X)//128])

    for hp in hpsearch.configs:
        for _ in range(3):
            model = build_model(2, train_X=train_X, train_y=train_y, **hp)
            with model.begin_training(train_X[:100], train_y[:100]) as (_, sgd):
                pass
            _, (model_data, train_acc, dev_acc) = train_epoch(model, sgd, hp,
                                                    train_X, train_y,
                                                    dev_X, dev_y,
                                                    device_id=-1 if not use_gpu else 0)
            print('0', dev_acc*100, train_acc*100, hp)
            hpsearch.enqueue(model_data, train_acc, dev_acc)
            hpsearch.temperature = 0.0
    print("Train")
    total = 0
    temperature = 0.0
    while True:
        for model, sgd, hp in hpsearch:
            _, (new_model, train_acc, dev_acc)  = train_epoch(model, sgd, hp,
                                                    train_X, train_y, dev_X, dev_y,
                                                    device_id=-1 if not use_gpu else 0,
                                                    temperature=hpsearch.temperature)
        hp = new_model[-1]
        print('%d,%d,%d:\t%.2f\t%.2f\t%.2f\t%d\t%.2f\t%.3f\t%d\t%d\t%.3f\t%.3f\t%.3f' % (
            total,
            hp['epochs'],
            hp['parent'],
            hpsearch.best_acc * 100,
            dev_acc * 100,
            train_acc * 100,
            int(hp['batch_size']),
            hp['dropout'],
            hp['learn_rate'],
            hp['width'],
            hp['depth'],
            hpsearch.temperature,
            hpsearch.queue[0][0],
            hpsearch.queue[-1][0]
        ))
        total += 1
        hpsearch.enqueue(new_model, train_acc, dev_acc)


if __name__ == '__main__':
    main()
