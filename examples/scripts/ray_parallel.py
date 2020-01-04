from typing import Optional
from pathlib import Path
from collections import defaultdict
import thinc
import thinc.config

try:
    import ray
except ImportError:
    class ray:
        @classmethod
        def remote(cls, func):
            return func


@ray.remote
class ParameterServer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def apply_gradients(self, worker_grads):
        summed_gradients = defaultdict(dict)
        for grads in worker_grads:
            for (node_id, name), grad in grads.items():

                if name in summed_gradients[node_id]:
                    summed_gradients[node_id][name] += grad
                else:
                    summed_gradients[node_id][name] = grad
        set_model_grads(model, summed_gradients)
        self.model.finish_update(self.optimizer)
        return get_model_weights(self.model)

    def get_weights(self):
        return get_model_weights(self.model)


@ray.remote
class DataWorker(object):
    def __init__(self, model):
        self.model = model
        self.data_iterator = iter(get_data_loader()[0])

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()


def get_model_weights(model):
    params = defaultdict(dict)
    for node in model.walk():
        for param in node._params:
            params[node.id][name] = node.get_param(name)
    return params


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
        Softmax()
    )

thinc.registry.create("loaders")
@thinc.registry.loaders("mnist.v1")
def load_mnist():
    import ml_datasets
    from thinc.util import to_categorical
    from thinc.backends import NumpyOps
    ops = NumpyOps
    # Load the data
    mnist_train, mnist_dev, _ = ml_datasets.mnist()
    train_X, train_Y = ops.unzip(mnist_train)
    train_Y = to_categorical(train_Y, nb_classes=10)
    dev_X, dev_Y = ops.unzip(mnist_dev)
    dev_Y = to_categorical(dev_Y, nb_classes=10)
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    print(len(train_X), len(train_Y))
    return (train_X, train_Y), (dev_X, dev_Y)


CONFIG = """
[training]
iterations = 200
num_workers = 1

[model]
@layers = "relu_relu_softmax.v1"
hidden_width = 128
dropout = 0.2

[optimizer]
@optimizers = "Adam.v1"

[dataset]
@loaders = "mnist.v1"
"""

def main(config_path: Optional[Path]=None):
    config_str = CONFIG if config_path is None else config_path.open().read()
    config = thinc.registry.make_from_config(thinc.config.Config().from_str(config_str))

    model = config["model"]
    optimizer = config["optimizer"]
    (train_X, train_Y), (dev_X, dev_Y) = config["dataset"]

    return None

    ray.init(ignore_reinit_error=True, object_store_memory=3000000000, num_cpus=3)
    ps = ParameterServer.remote(1e-2)
    workers = [DataWorker.remote() for i in range(num_workers)]

    print("Running synchronous parameter server training.")
    current_weights = ps.get_weights.remote()
    for i in range(iterations):
    	gradients = [
        	worker.compute_gradients.remote(current_weights) for worker in workers
    	]
    	# Calculate update after all gradients are available.
    	current_weights = ps.apply_gradients.remote(*gradients)

    	if i % 10 == 0:
            # Evaluate the current model.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))
    # Clean up Ray resources and processes before the next example.
    ray.shutdown()


if __name__ == "__main__":
    import typer
    typer.run(main)
