"""
Compare tagging speed for LSTM, using dummy data.

Results on CPU laptop:

PyTorchLSTM.v0:
Predicted 39017 4.033804399892688 Ys[0] 0.05000001 5.551115e-17

LSTM (NumpyOps):
Predicted 39018 13.174870599992573 Ys[0] 0.05000001 5.551115e-17

So PyTorch is 3x faster currently.
"""
import typer
from timeit import default_timer as timer
import numpy.random
from timeit import default_timer as timer
from typing import List, cast
from thinc.types import Array2d, Padded
import thinc.api
from thinc.api import Model, Config, registry, Embed, LSTM, Softmax
from thinc.api import chain, list2padded, with_array, to_categorical
from thinc.api import minibatch


CONFIG = """
[data]
n_samples = 1000
n_tags = 20
n_vocab = 10000
length_mean = 40
length_variance = 10

[common]
width = 300

[model]
@layers = "LSTMTagger.v0"

[model.embed]
@layers = "Embed.v0"
nO = ${common:width}
nV = ${data:n_vocab}

[model.encode]
@layers = "LSTM.v0"
nO = ${common:width}
depth = 2

[model.predict]
@layers = "Softmax.v0"
nO = ${data:n_tags}
"""

@registry.layers("LSTMTagger.v0")
def build_tagger(
        embed: Model[Array2d, Array2d],
        encode: Model[Padded, Padded],
        predict: Model[Array2d, Array2d]
) -> Model[List[Array2d], Padded]:
    model = chain(
        list2padded(),
        with_array(embed),
        encode,
        with_array(predict),
    )
    model.set_ref("embed", embed)
    model.set_ref("encode", encode)
    model.set_ref("predict", model.layers[-1])
    return model


def get_dummy_data(n_samples, n_tags, n_vocab, length_mean, length_variance):
    Xs = []
    Ys = []
    for _ in range(n_samples):
        length = numpy.random.normal(size=1, scale=length_variance) + length_mean
        shape = (max(1, int(length)),)
        X = numpy.random.uniform(0, n_vocab-1, shape)
        Y = numpy.random.uniform(0, n_tags-1, shape)
        assert X.size, length
        assert Y.size, length
        Xs.append(X.reshape((-1, 1)).astype("i"))
        Ys.append(to_categorical(Y.astype("i")))
    return Xs, Ys

def run_forward(model, batches):
    Ys = []
    i = 0
    for batch in batches:
        Ys.append(model.predict(batch))
    return Ys

def main():
    #thinc.api.set_current_ops(thinc.api.JaxOps())
    numpy.random.seed(0)
    C = registry.make_from_config(Config().from_str(CONFIG))
    model = C["model"]
    X, Y = get_dummy_data(**C["data"])
    print("Begin init", len(X))
    model.initialize(X=X[:5])
    print("Copy data to jax")
    X = [model.ops.asarray(x) for x in X]
    print("Minibatch and pad")
    X = [model.layers[0].predict(batch) for batch in minibatch(X, size=128)]
    model.layers.pop(0)
    print("Start")
    start_time = timer()
    Ys = run_forward(model, X)
    end_time = timer()
    print("Predicted", sum(sum(x.lengths) for x in X), end_time-start_time)
    print("Ys[0]", Ys[0].data.mean(), Ys[0].data.var())


if __name__ == "__main__":
    main()
