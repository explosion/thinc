"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
from thinc.api import Model, chain, Relu, Softmax, Adam
import ml_datasets
from wasabi import msg
from tqdm import tqdm
import typer
from thinc.backends._ray_remote_params import RayProxy, SharedOptimizer
import ray


@ray.remote
class Worker:
    def __init__(self, i, n_workers):
        self.i = i
        self.n_workers = n_workers
        self.model = None
        self.optimizer = None
        self.train_data = None
        self.dev_data = None

    def add_model(self, n_hidden, dropout):
        # Define the model
        self.model = chain(
            Relu(nO=n_hidden, dropout=dropout),
            Relu(nO=n_hidden, dropout=dropout),
            Softmax(),
        )

    def add_data(self, batch_size):
        # Load the data
        (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
        shard_size = len(train_X) // self.n_workers
        shard_start = self.i * shard_size
        shard_end = shard_start + shard_size
        self.train_data = self.model.ops.multibatch(
            batch_size,
            train_X[shard_start : shard_end],
            train_Y[shard_start : shard_end],
            shuffle=True
        )
        self.dev_data = self.model.ops.multibatch(batch_size, dev_X, dev_Y)
        # Set any missing shapes for the model.
        self.model.initialize(X=train_X[:5], Y=train_Y[:5])

    def set_proxy(self, connection):
        proxy = RayProxy(connection)
        for node in self.model.walk():
            for name in node.param_names:
                proxy.set_param(node.id, name, node.get_param(name))
            node._params.proxy = proxy

    def train_epoch(self):
        for X, Y in self.train_data:
            Yh, backprop = self.model.begin_update(X)
            backprop(Yh - Y)
            #model.finish_update(self.optimizer)

    def evaluate(self):
        correct = 0
        total = 0
        for X, Y in self.dev_data:
            Yh = self.model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        return correct / total


def main(
    n_hidden: int = 256, dropout: float = 0.2, n_iter: int = 10, batch_size: int = 128,
    n_epoch: int=10, quorum: int=2, n_workers: int=1
):
    ray.init(num_cpus=3)
    workers = []
    conn = ray.remote(SharedOptimizer).remote(quorum, Adam(0.001))
    print("Create workers")
    for i in range(n_workers):
        worker = Worker.remote(i, n_workers)
        ray.get(worker.add_model.remote(n_hidden, dropout))
        ray.get(worker.add_data.remote(batch_size))
        ray.get(worker.set_proxy.remote(conn))
        workers.append(worker)
    for i in range(n_epoch):
        futures = []
        for worker in workers:
            futures.append(worker.train_epoch.remote())
        ray.get(futures)
        print(i, ray.get(workers[0].evaluate.remote()))


if __name__ == "__main__":
    typer.run(main)
