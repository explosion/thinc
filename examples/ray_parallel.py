"""This script is still a work in progress: using Ray to implement parallel
training. The example is based off one of Ray's tutorials:
https://ray.readthedocs.io/en/latest/auto_examples/plot_parameter_server.html
"""
# pip install thinc ml_datasets typer
from typing import Optional
from pathlib import Path
from collections import defaultdict
import thinc
import thinc.config
from thinc.util import fix_random_seed
import ml_datasets
import typer

try:
    import ray
except ImportError:

    class ray:
        @classmethod
        def remote(cls, func):
            return func


MNIST = ml_datasets.mnist()


@ray.remote
class ParameterServer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def apply_gradients(self, *worker_grads):
        summed_gradients = defaultdict(dict)
        for grads in worker_grads:
            for node_id, node_grads in grads.items():
                for name, grad in node_grads.items():
                    if name in summed_gradients[node_id]:
                        summed_gradients[node_id][name] += grad
                    else:
                        summed_gradients[node_id][name] = grad.copy()
        set_model_grads(self.model, summed_gradients)
        self.model.finish_update(self.optimizer)
        return get_model_weights(self.model)

    def get_weights(self):
        return get_model_weights(self.model)


@ray.remote
class DataWorker:
    def __init__(self, model, batch_size=128, seed=0):
        self.model = model
        fix_random_seed(seed)
        self.data_iterator = iter(get_data_loader(batch_size)[0])
        self.batch_size = batch_size

    def compute_gradients(self, weights):
        set_model_weights(self.model, weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader(self.batch_size)[0])
            data, target = next(self.data_iterator)

        guesses, backprop = self.model(data, is_train=True)
        backprop((guesses - target) / target.shape[0])
        return get_model_grads(self.model)


def get_model_weights(model):
    params = defaultdict(dict)
    for node in model.walk():
        for name in node.params_names:
            if node.has_param(name):
                params[node.id][name] = node.get_param(name)
    return params


def get_model_grads(model):
    grads = defaultdict(dict)
    for node in model.walk():
        for name in node.grad_names:
            grads[node.id][name] = node.get_grad(name)
    return grads


def set_model_weights(model, params):
    for node in model.walk():
        for name, param in params[node.id].items():
            node.set_param(name, param)


def set_model_grads(model, grads):
    for node in model.walk():
        for name, grad in grads[node.id].items():
            node.set_grad(name, grad)


@thinc.registry.layers("relu_relu_softmax.v1")
def make_relu_relu_softmax(hidden_width: int, dropout: float):
    from thinc.layers import chain, ReLu, Softmax

    return chain(
        ReLu(hidden_width, dropout=dropout),
        ReLu(hidden_width, dropout=dropout),
        Softmax(),
    )


def get_data_loader(batch_size):
    from thinc.util import get_shuffled_batches

    (train_X, train_Y), (dev_X, dev_Y) = MNIST

    train_batches = get_shuffled_batches(train_X, train_Y, batch_size)
    dev_batches = get_shuffled_batches(dev_X, dev_Y, batch_size)
    return train_batches, dev_batches


def evaluate(model, batch_size):
    from thinc.util import evaluate_model_on_arrays

    dev_X, dev_Y = MNIST[1]
    return evaluate_model_on_arrays(model, dev_X, dev_Y, batch_size)


CONFIG = """
[ray]
num_workers = 1
object_store_memory = 3000000000
num_cpus = 2

[training]
iterations = 200
batch_size = 128

[evaluation]
batch_size = 256
frequency = 10

[model]
@layers = "relu_relu_softmax.v1"
hidden_width = 128
dropout = 0.2

[optimizer]
@optimizers = "Adam.v1"
"""


def main(config_path: Optional[Path] = None):
    # You can edit the CONFIG string within the file, or copy it out to
    # a separate file and pass in the path.
    config_str = CONFIG if config_path is None else config_path.open().read()
    # The make_from_config function constructs objects for you, whenever
    # you have blocks with an @ key. For instance, in the optimizer block,
    # we write @optimizers = "Adam.v1". This tells Thinc to use the optimizers
    # registry to fetch the "Adam.v1" function. You can register your own
    # functions as well, and build up trees of objects.
    config = thinc.registry.make_from_config(thinc.config.Config().from_str(config_str))
    # Here we have the model and optimizer, built for us by the registry.
    model = config["model"]
    optimizer = config["optimizer"]
    (train_X, train_Y), (dev_X, dev_Y) = MNIST
    # We didn't specify all the dimensions in the model, so we need to pass in
    # a batch of data to finish initialization. This lets Thinc infer the missing
    # shapes.
    model.initialize(X=train_X[:5], Y=train_Y[:5])

    # Now the Ray stuff...
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=config["ray"]["object_store_memory"],
        num_cpus=config["ray"]["num_cpus"],
    )
    ps = ParameterServer.remote(model, optimizer)
    workers = []
    for i in range(config["ray"]["num_workers"]):
        # The Ray tutorial didn't tell me to set a different random seed for
        # the workers, but to me it makes sense? Otherwise it seems the workers
        # will iterate over the batches in the same order, which seems wrong?
        workers.append(
            DataWorker.remote(
                model, batch_size=config["training"]["batch_size"], seed=i
            )
        )

    print("Running synchronous parameter server training.")
    current_weights = ps.get_weights.remote()
    for i in range(config["training"]["iterations"]):
        gradients = [
            worker.compute_gradients.remote(current_weights) for worker in workers
        ]
        # Calculate update after all gradients are available.
        current_weights = ps.apply_gradients.remote(*gradients)

        if i % config["evaluation"]["frequency"] == 0:
            # Evaluate the current model.
            set_model_weights(model, ray.get(current_weights))
            accuracy = evaluate(model, config["evaluation"]["batch_size"])
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))
    # Clean up Ray resources and processes before the next example.
    ray.shutdown()


if __name__ == "__main__":
    typer.run(main)
