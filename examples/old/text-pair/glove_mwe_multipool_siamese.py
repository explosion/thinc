# Warning: The code here might be broken --- the results don't seem right
import sys
import plac
from pathlib import Path
from srsly import cloudpickle as pickle
from thinc.v2v import Model, Maxout
from thinc.t2v import Pooling, mean_pool, max_pool
from thinc.i2v import HashEmbed, StaticVectors
from thinc.t2t import ExtractWindow
from thinc.neural._classes.difference import Siamese, CauchySimilarity
from thinc.misc import LayerNorm as LN
from thinc.misc import Residual
from thinc.neural.ops import CupyOps
from thinc.neural.util import require_gpu
from thinc.api import layerize, with_getitem, flatten_add_lengths
from thinc.api import add, chain, clone, concatenate, get_word_ids
from thinc.extra import datasets
from thinc.extra.load_nlp import get_spacy


try:
    import spacy

    spacy.load("en_vectors_web_lg")
except (ImportError, OSError):
    print("Missing dependency: spacy. Try:")
    print("pip install spacy")
    print("python -m spacy download en_vectors_web_lg")
    sys.exit(1)


epoch_train_acc = 0.0
epoch = 0


print("Warning: The code here might be broken --- the results don't seem right")


def track_progress(**context):
    """Print training progress. Called after each epoch."""
    model = context["model"]
    train_X = context["train_X"]
    dev_X = context["dev_X"]
    dev_y = context["dev_y"]
    n_train = len(train_X)
    trainer = context["trainer"]

    def each_epoch():
        global epoch_train_acc, epoch
        with model.use_params(trainer.optimizer.averages):
            avg_acc = model.evaluate_logloss(dev_X, dev_y)
        stats = (avg_acc, float(epoch_train_acc) / n_train, trainer.dropout)
        print("%.3f dev acc, %.3f train acc, %.4f drop" % stats)
        epoch_train_acc = 0.0
        epoch += 1

    return each_epoch


def preprocess(ops, nlp, rows, get_ids):
    """Parse the texts with spaCy. Make one-hot vectors for the labels."""
    Xs = []
    ys = []
    for (text1, text2), label in rows:
        Xs.append((get_ids([nlp(text1)])[0], get_ids([nlp(text2)])[0]))
        ys.append(label)
    return Xs, ops.asarray(ys, dtype="float32")


@layerize
def logistic(X, drop=0.0):
    ops = Model.ops
    y = 1.0 / (1.0 + ops.xp.exp(-X))

    def backward(dy, sgd=None):
        return dy * y * (1 - y)

    return y, backward


@plac.annotations(
    dataset=("Dataset to load"),
    width=("Width of the hidden layers", "option", "w", int),
    depth=("Depth of the hidden layers", "option", "d", int),
    min_batch_size=("Minimum minibatch size during training", "option", "b", int),
    max_batch_size=("Maximum minibatch size during training", "option", "B", int),
    L2=("L2 penalty", "option", "L", float),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
    use_gpu=("Whether to use GPU", "flag", "G", bool),
    nb_epoch=("Number of epochs", "option", "i", int),
    pieces=("Number of pieces for maxout", "option", "p", int),
    out_loc=("File to save the model", "option", "o"),
    quiet=("Don't print the progress bar", "flag", "q"),
    pooling=("Which pooling to use", "option", "P", str),
    job_id=("Job ID for Neptune", "option", "J"),
    rest_api_url=("REST API URL", "option", "R"),
    ws_api_url=("WS API URL", "option", "W"),
)
def main(
    dataset="quora",
    width=200,
    depth=2,
    min_batch_size=1,
    max_batch_size=512,
    dropout=0.2,
    dropout_decay=0.0,
    pooling="mean+max",
    nb_epoch=5,
    pieces=3,
    L2=0.0,
    use_gpu=False,
    out_loc=None,
    quiet=False,
    job_id=None,
    ws_api_url=None,
    rest_api_url=None,
):
    require_gpu()
    cfg = dict(locals())

    if out_loc:
        out_loc = Path(out_loc)
        if not out_loc.parent.exists():
            raise IOError("Can't open output location: %s" % out_loc)
    print(cfg)
    if pooling == "mean+max":
        pool_layer = Pooling(mean_pool, max_pool)
    elif pooling == "mean":
        pool_layer = mean_pool
    elif pooling == "max":
        pool_layer = max_pool
    else:
        raise ValueError("Unrecognised pooling", pooling)

    print("Load spaCy")
    nlp = get_spacy("en_vectors_web_lg")

    if use_gpu:
        Model.ops = CupyOps()

    print("Construct model")
    # Bind operators for the scope of the block:
    # * chain (>>): Compose models in a 'feed forward' style,
    # i.e. chain(f, g)(x) -> g(f(x))
    # * clone (**): Create n copies of a model, and chain them, i.e.
    # (f ** 3)(x) -> f''(f'(f(x))), where f, f' and f'' have distinct weights.
    # * concatenate (|): Merge the outputs of two models into a single vector,
    # i.e. (f|g)(x) -> hstack(f(x), g(x))
    Model.lsuv = True
    # Model.ops = CupyOps()
    with Model.define_operators({">>": chain, "**": clone, "|": concatenate, "+": add}):
        mwe_encode = ExtractWindow(nW=1) >> LN(
            Maxout(width, drop_factor=0.0, pieces=pieces)
        )

        sent2vec = (
            flatten_add_lengths
            >> with_getitem(
                0,
                (HashEmbed(width, 3000) | StaticVectors("en_vectors_web_lg", width))
                >> LN(Maxout(width, width * 2))
                >> Residual(mwe_encode) ** depth,
            )  # : word_ids{T}
            >> Pooling(mean_pool, max_pool)
            >> Residual(LN(Maxout(width * 2, pieces=pieces), nO=width * 2)) ** 2
            >> logistic
        )
        model = Siamese(sent2vec, CauchySimilarity(width * 2))

    print("Read and parse data: %s" % dataset)
    if dataset == "quora":
        train, dev = datasets.quora_questions()
    elif dataset == "snli":
        train, dev = datasets.snli()
    elif dataset == "stackxc":
        train, dev = datasets.stack_exchange()
    elif dataset in ("quora+snli", "snli+quora"):
        train, dev = datasets.quora_questions()
        train2, dev2 = datasets.snli()
        train.extend(train2)
        dev.extend(dev2)
    else:
        raise ValueError("Unknown dataset: %s" % dataset)
    get_ids = get_word_ids(Model.ops)
    train_X, train_y = preprocess(model.ops, nlp, train, get_ids)
    dev_X, dev_y = preprocess(model.ops, nlp, dev, get_ids)

    with model.begin_training(train_X[:10000], train_y[:10000], **cfg) as (
        trainer,
        optimizer,
    ):
        # Pass a callback to print progress. Give it all the local scope,
        # because why not?
        trainer.each_epoch.append(track_progress(**locals()))
        trainer.batch_size = min_batch_size
        batch_size = float(min_batch_size)
        print("Accuracy before training", model.evaluate_logloss(dev_X, dev_y))
        print("Train")
        global epoch_train_acc
        n_iter = 0

        for X, y in trainer.iterate(train_X, train_y, progress_bar=not quiet):
            # Slightly useful trick: Decay the dropout as training proceeds.
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            assert yh.shape == y.shape, (yh.shape, y.shape)

            assert (yh >= 0.0).all(), yh
            train_acc = ((yh >= 0.5) == (y >= 0.5)).sum()
            loss = model.ops.xp.abs(yh - y).mean()
            epoch_train_acc += train_acc
            backprop(yh - y, optimizer)
            n_iter += 1

            # Slightly useful trick: start with low batch size, accelerate.
            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001
        if out_loc:
            out_loc = Path(out_loc)
            print("Saving to", out_loc)
            with out_loc.open("wb") as file_:
                pickle.dump(model, file_, -1)


if __name__ == "__main__":
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats

        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(100)
