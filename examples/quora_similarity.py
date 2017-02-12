from __future__ import unicode_literals, print_function
import plac
import spacy
from pathlib import Path
import dill as pickle

from thinc.neural import Model, Softmax, Maxout
from thinc.neural import ExtractWindow
#from thinc.neural.vecs2vec import Pooling, mean_pool, max_pool
from thinc.neural.pooling import Pooling, mean_pool, max_pool
from thinc.neural._classes.spacy_vectors import SpacyVectors, get_word_ids

from thinc.neural.util import to_categorical

from thinc.api import flatten_add_lengths, with_getitem
from thinc.api import chain, clone, concatenate, Arg

from thinc.extra import datasets
from thinc.extra.load_nlp import get_spacy



epoch_train_acc = 0.
def track_progress(**context):
    '''Print training progress. Called after each epoch.'''
    model = context['model']
    train_X = context['train_X']
    dev_X = context['dev_X']
    dev_y = context['dev_y']
    n_train = len(train_X)
    trainer = context['trainer']
    def each_epoch():
        global epoch_train_acc
        acc = model.evaluate(dev_X, dev_y)
        with model.use_params(trainer.optimizer.averages):
            avg_acc = model.evaluate(dev_X, dev_y)
        stats = (acc, avg_acc, float(epoch_train_acc) / n_train, trainer.dropout)
        print("%.3f (%.3f) dev acc, %.3f train acc, %.4f drop" % stats)
        epoch_train_acc = 0.
    return each_epoch


def preprocess(ops, nlp, rows):
    '''Parse the texts with spaCy. Make one-hot vectors for the labels.'''
    Xs = []
    ys = []
    for (text1, text2), label in rows:
        Xs.append((nlp(text1), nlp(text2)))
        ys.append(label)
    return Xs, to_categorical(ops.asarray(ys))


@plac.annotations(
    loc=("Location of Quora data"),
    width=("Width of the hidden layers", "option", "w", int),
    depth=("Depth of the hidden layers", "option", "d", int),
    min_batch_size=("Minimum minibatch size during training", "option", "b", int),
    max_batch_size=("Maximum minibatch size during training", "option", "B", int),
    dropout=("Dropout rate", "option", "D", float),
    dropout_decay=("Dropout decay", "option", "C", float),
    use_gpu=("Whether to use GPU", "flag", "G", bool),
    nb_epoch=("Number of epochs", "option", "i", int),
    pieces=("Number of pieces for maxout", "option", "p", int),
    out_loc=("File to save the model", "option", "o"),
    quiet=("Don't print the progress bar", "flag", "q")
)
def main(loc=None, width=128, depth=2, min_batch_size=1, max_batch_size=256,
         dropout=0.5, dropout_decay=1e-5,
         nb_epoch=20, pieces=3, use_gpu=False, out_loc=None, quiet=False):
    cfg = dict(locals())
    if out_loc:
        out_loc = Path(out_loc)
        if not out_loc.parent.exists():
            raise IOError("Can't open output location: %s" % out_loc)
    cfg = dict(locals())
    print(cfg)

    print("Load spaCy")
    nlp = get_spacy('en')

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
    with Model.define_operators({'>>': chain, '**': clone, '|': concatenate}):
        # Important trick: text isn't like images, and the best way to use
        # convolution is different. Don't use pooling-over-time. Instead,
        # use the window to compute one vector per word, and do this N deep.
        # In the first layer, we adjust each word vector based on the two
        # surrounding words --- this gives us essentially trigram vectors.
        # In the next layer, we have a trigram of trigrams --- so we're
        # conditioning on information from a five word slice. The third layer
        # gives us 7 words. This is like the BiLSTM insight: we're not trying
        # to learn a vector for the whole sentence in this step. We're just
        # trying to learn better, position-sensitive word features. This simple
        # convolution step is much more efficient than BiLSTM, and can be
        # computed in parallel for every token in the batch.
        mwe_encode = ExtractWindow(nW=1) >> Maxout(width, width*3, pieces=pieces)
        # Comments indicate the output type and shape at each step of the pipeline.
        # * B: Number of sentences in the batch
        # * T: Total number of words in the batch
        # (i.e. sum(len(sent) for sent in batch))
        # * W: Width of the network (input hyper-parameter)
        # * ids: ID for each word (integers).
        # * lengths: Number of words in each sentence in the batch (integers)
        # * floats: Standard dense vector.
        # (Dimensions annotated in curly braces.)
        sent2vec = ( # List[spacy.token.Doc]{B}
            get_word_ids            # : List[ids]{B}
            >> flatten_add_lengths  # : (ids{T}, lengths{B})
            >> with_getitem(0,      # : word_ids{T}
                # This class integrates a linear projection layer, and loads
                # static embeddings (by default, GloVe common crawl).
                SpacyVectors('en', width) # : floats{T, W}
                >> mwe_encode ** depth   # : floats{T, W}
            ) # : (floats{T, W}, lengths{B})
            # Useful trick: Why choose between max pool and mean pool?
            # We may as well have both representations.
            >> Pooling(mean_pool, max_pool) # : floats{B, 2*W}
        )
        model = (
            ((Arg(0) >> sent2vec) | (Arg(1) >> sent2vec)) # : floats{B, 4*W}
            >> Maxout(width, width*4, pieces=pieces) # : floats{B, W}
            >> Softmax(2, width) # : floats{B, 2}
        )


    print("Read and parse quora data")
    train, dev = datasets.quora_questions(loc)
    train_X, train_y = preprocess(model.ops, nlp, train)
    dev_X, dev_y = preprocess(model.ops, nlp, dev)
    assert len(dev_y.shape) == 2
    print("Initialize with data (LSUV)")
    with model.begin_training(train_X[:5000], train_y[:5000], **cfg) as (trainer, optimizer):
        # Pass a callback to print progress. Give it all the local scope,
        # because why not?
        trainer.each_epoch.append(track_progress(**locals()))
        trainer.batch_size = min_batch_size
        batch_size = float(min_batch_size)
        print("Accuracy before training", model.evaluate(dev_X, dev_y))
        print("Train")
        global epoch_train_acc
        for X, y in trainer.iterate(train_X, train_y, progress_bar=not quiet):
            # Slightly useful trick: Decay the dropout as training proceeds.
            yh, backprop = model.begin_update(X, drop=trainer.dropout)
            # No auto-diff: Just get a callback and pass the data through.
            # Hardly a hardship, and it means we don't have to create/maintain
            # a computational graph. We just use closures.
            backprop(yh-y, optimizer)

            epoch_train_acc += (yh.argmax(axis=1) == y.argmax(axis=1)).sum()

            # Slightly useful trick: start with low batch size, accelerate.
            trainer.batch_size = min(int(batch_size), max_batch_size)
            batch_size *= 1.001
        if out_loc:
            out_loc = Path(out_loc)
            print('Saving to', out_loc)
            with out_loc.open('wb') as file_:
                pickle.dump(model, file_, -1)


if __name__ == '__main__':
    if 1:
        plac.call(main)
    else:
        import cProfile
        import pstats
        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats(100)
