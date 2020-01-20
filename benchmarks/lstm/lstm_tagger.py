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
from thinc.backends import jax_jit
import jax.tree_util

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
@layers = "LSTMTagger.v0"

[model.embed]
@layers = "Embed.v0"
nO = ${common:width}
nV = ${data:n_vocab}

[model.encode]
@layers = "LSTM.v0"
nO = ${common:width}
nI = ${common:width}
depth = 1

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
        #with_array(predict),
    )
    model.set_ref("embed", embed)
    model.set_ref("encode", encode)
    model.set_ref("predict", model.layers[-1])
    return model


def get_dummy_data(n_samples, n_tags, n_vocab, length_mean, length_variance):
    Xs = []
    Ys = []
    for _ in range(n_samples):
        #length = numpy.random.normal(size=1, scale=length_variance) + length_mean
        length = length_mean
        shape = (max(1, int(length)),)
        X = numpy.random.uniform(0, n_vocab-1, shape)
        Y = numpy.random.uniform(0, n_tags-1, shape)
        assert X.size, length
        assert Y.size, length
        Xs.append(X.reshape((-1, 1)).astype("i"))
        Ys.append(to_categorical(Y.astype("i")))
    return Xs, Ys

jax.tree_util.register_pytree_node(
    Padded,
    lambda pad: ((pad.data, pad.size_at_t, pad.lengths, pad.indices), None),
    lambda info, values: Padded(*values)
)

def run_forward(model, Xs):
    total = 0.
    for batch in Xs:
        Y = model.predict(batch)
        total += Y.data.sum()
    return float(total)

def set_backend(name, gpu_id):
    global CONFIG
    if name == "jax":
        thinc.api.set_current_ops(thinc.api.JaxOps())
        CONFIG = CONFIG.replace("PyTorch", "")
    else:
        if gpu_id == -1:
            thinc.api.set_current_ops(thinc.api.NumpyOps())
        else:
            thinc.api.set_current_ops(thinc.api.CupyOps())
        CONFIG = CONFIG.replace("LSTM.v0", "PyTorchLSTM.v0")


def main(jax: bool=False, pytorch: bool=False, gpu_id: int=-1):
    global CONFIG
    thinc.api.fix_random_seed(0)
    if gpu_id >= 0:
        thinc.api.require_gpu(gpu_id)
        print("Set GPU", gpu_id)
    backends = {"jax": jax, "pytorch": pytorch}
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
        X = [model.layers[0].predict(batch) for batch in minibatch(X, size=16)]
        model.layers.pop(0)
        print("Start")
        start_time = timer()
        total = run_forward(model, X)
        end_time = timer()
        print(name, n_words, end_time-start_time)


if __name__ == "__main__":
    typer.run(main)
