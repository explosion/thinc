import tqdm
from thinc.i2v import StaticVectors, HashEmbed
from thinc.v2v import Model, Affine, Maxout, Softmax
from thinc.t2t import ParametricAttention
from thinc.t2t import MultiHeadedAttention, prepare_self_attention
from thinc.t2v import Pooling, mean_pool
from thinc.misc import LayerNorm as LN
from thinc.misc import Residual
from thinc.extra import datasets
from thinc.extra.load_nlp import register_vectors
from thinc.neural.util import to_categorical, require_gpu
from thinc.api import layerize, chain, concatenate, clone
from thinc.api import foreach, with_flatten, flatten_add_lengths
from thinc.misc import FeatureExtracter
import spacy
from spacy.attrs import ORTH, LOWER, SHAPE, PREFIX, SUFFIX, ID
from spacy.util import compounding, fix_random_seed


@layerize
def get_sents(docs, drop=0.0):
    sents = [list(doc.sents) for doc in docs]
    return sents, None


def build_model(nr_class, width, depth, conv_depth, vectors_name, **kwargs):
    with Model.define_operators({"|": concatenate, ">>": chain, "**": clone}):
        embed = (
            HashEmbed(width, 5000, column=1)
            | StaticVectors(vectors_name, width, column=5)
            | HashEmbed(width // 2, 750, column=2)
            | HashEmbed(width // 2, 750, column=3)
            | HashEmbed(width // 2, 750, column=4)
        ) >> LN(Maxout(width))

        sent2vec = (
            with_flatten(embed)
            >> Residual(
                prepare_self_attention(Affine(width * 3, width), nM=width, nH=4)
                >> MultiHeadedAttention()
                >> with_flatten(Maxout(width, width, pieces=3))
            )
            >> flatten_add_lengths
            >> ParametricAttention(width, hard=False)
            >> Pooling(mean_pool)
            >> Residual(LN(Maxout(width)))
        )

        model = (
            foreach(sent2vec, drop_factor=2.0)
            >> Residual(
                prepare_self_attention(Affine(width * 3, width), nM=width, nH=4)
                >> MultiHeadedAttention()
                >> with_flatten(LN(Affine(width, width)))
            )
            >> flatten_add_lengths
            >> ParametricAttention(width, hard=False)
            >> Pooling(mean_pool)
            >> Residual(LN(Maxout(width))) ** 2
            >> Softmax(nr_class)
        )
    model.lsuv = False
    return model


def main(use_gpu=False, nb_epoch=100):
    fix_random_seed(0)
    if use_gpu:
        require_gpu()
    train, test = datasets.imdb(limit=2000)
    print("Load data")
    train_X, train_y = zip(*train)
    test_X, test_y = zip(*test)
    train_y = Model.ops.asarray(to_categorical(train_y, nb_classes=2))
    test_y = Model.ops.asarray(to_categorical(test_y, nb_classes=2))

    nlp = spacy.load("en_vectors_web_lg")
    nlp.add_pipe(nlp.create_pipe("sentencizer"), first=True)
    register_vectors(Model.ops, nlp.vocab.vectors.name, nlp.vocab.vectors.data)

    preprocessor = FeatureExtracter([ORTH, LOWER, PREFIX, SUFFIX, SHAPE, ID])
    train_X = [preprocessor(list(doc.sents)) for doc in tqdm.tqdm(nlp.pipe(train_X))]
    test_X = [preprocessor(list(doc.sents)) for doc in tqdm.tqdm(nlp.pipe(test_X))]

    dev_X = train_X[-1000:]
    dev_y = train_y[-1000:]
    train_X = train_X[:-1000]
    train_y = train_y[:-1000]
    print("Parse data")
    n_sent = sum([len(list(sents)) for sents in train_X])
    print("%d sentences" % n_sent)

    model = build_model(
        2,
        vectors_name=nlp.vocab.vectors.name,
        width=128,
        conv_depth=2,
        depth=2,
        train_X=train_X,
        train_y=train_y,
    )
    with model.begin_training(train_X[:100], train_y[:100]) as (trainer, optimizer):
        epoch_loss = [0.0]

        def report_progress():
            with model.use_params(optimizer.averages):
                print(
                    epoch_loss[-1],
                    epoch_var[-1],
                    model.evaluate(dev_X, dev_y),
                    trainer.dropout,
                )
            epoch_loss.append(0.0)
            epoch_var.append(0.0)

        trainer.each_epoch.append(report_progress)
        batch_sizes = compounding(64, 64, 1.01)
        trainer.dropout = 0.3
        trainer.batch_size = int(next(batch_sizes))
        trainer.dropout_decay = 0.0
        trainer.nb_epoch = nb_epoch
        # optimizer.alpha = 0.1
        # optimizer.max_grad_norm = 10.0
        # optimizer.b1 = 0.0
        # optimizer.b2 = 0.0
        epoch_var = [0.0]
        for X, y in trainer.iterate(train_X, train_y):
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            losses = ((yh - y) ** 2.0).sum(axis=1) / y.shape[0]
            epoch_var[-1] += losses.var()
            loss = losses.mean()
            backprop((yh - y) / yh.shape[0], optimizer)
            epoch_loss[-1] += loss
            trainer.batch_size = int(next(batch_sizes))
        with model.use_params(optimizer.averages):
            print("Avg dev.: %.3f" % model.evaluate(dev_X, dev_y))


if __name__ == "__main__":
    main()
