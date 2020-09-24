"""
Compare tagging speed for LSTM, using dummy data.

Results on CPU laptop:

PyTorchLSTM.v1:
Predicted 39017 4.033804399892688 Ys[0] 0.05000001 5.551115e-17

LSTM (NumpyOps):
Predicted 39018 13.174870599992573 Ys[0] 0.05000001 5.551115e-17

So PyTorch is 3x faster currently.
"""
from typing import List
import typer
import numpy.random
from timeit import default_timer as timer
from thinc.api import Model, Config, registry, chain, list2padded, with_array
from thinc.api import to_categorical, set_current_ops
from thinc.api import NumpyOps, CupyOps, fix_random_seed, require_gpu
from thinc.types import Array2d, Padded

CONFIG = """
[data]
n_samples = 100000
n_tags = 20
n_vocab = 10000
length_mean = 40
length_variance = 1

[common]
width = 300

[model]
@layers = "LSTMTagger.v1"

[model.embed]
@layers = "Embed.v1"
nO = ${common:width}
nV = ${data:n_vocab}

[model.encode]
@layers = "LSTM.v1"
nO = ${common:width}
nI = ${common:width}
depth = 1

[model.predict]
@layers = "Softmax.v1"
nO = ${data:n_tags}
"""


@registry.layers("LSTMTagger.v1")
def build_tagger(
    embed: Model[Array2d, Array2d],
    encode: Model[Padded, Padded],
    predict: Model[Array2d, Array2d],
) -> Model[List[Array2d], Padded]:
    model = chain(
        list2padded(),
        with_array(embed),
        encode,
        # with_array(predict),
    )
    model.set_ref("embed", embed)
    model.set_ref("encode", encode)
    model.set_ref("predict", model.layers[-1])
    return model


def get_dummy_data(n_samples, n_tags, n_vocab, length_mean, length_variance):
    Xs = []
    Ys = []
    for _ in range(n_samples):
        # length = numpy.random.normal(size=1, scale=length_variance) + length_mean
        length = length_mean
        shape = (max(1, int(length)),)
        X = numpy.random.uniform(0, n_vocab - 1, shape)
        Y = numpy.random.uniform(0, n_tags - 1, shape)
        assert X.size, length
        assert Y.size, length
        Xs.append(X.reshape((-1, 1)).astype("i"))
        Ys.append(to_categorical(Y.astype("i")))
    return Xs, Ys


def run_forward(model, Xs):
    total = 0.0
    for batch in Xs:
        Y = model.predict(batch)
        total += Y.data.sum()
    return float(total)


def set_backend(name, gpu_id):
    if gpu_id == -1:
        set_current_ops(NumpyOps())
    else:
        set_current_ops(CupyOps())
    CONFIG = CONFIG.replace("LSTM.v1", "PyTorchLSTM.v1")


def main(pytorch: bool = False, gpu_id: int = -1):
    global CONFIG
    fix_random_seed(0)
    if gpu_id >= 0:
        require_gpu(gpu_id)
        print("Set GPU", gpu_id)
    backends = {"pytorch": pytorch}
    for name, use_backend in backends.items():
        if not use_backend:
            print(f"Skipping {name}")
            continue
        set_backend(name, gpu_id)
        C = registry.make_from_config(Config().from_str(CONFIG))
        model = C["model"]
        X, Y = get_dummy_data(**C["data"])
        print("Copy to device")
        X = [model.ops.asarray(x) for x in X]
        Y = [model.ops.asarray(y) for y in Y]
        print("Begin init", len(X))
        model.initialize(X=X[:5])
        print("Pre-batch")
        n_words = sum(len(x) for x in X)
        X = [model.layers[0].predict(batch) for batch in model.ops.minibatch(16, X)]
        model.layers.pop(0)
        print("Start")
        start_time = timer()
        end_time = timer()
        print(name, n_words, end_time - start_time)


if __name__ == "__main__":
    typer.run(main)
