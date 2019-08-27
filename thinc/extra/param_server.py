import random
from ..neural.util import minibatch


def dummy_decorator(*args, **kwargs):
    if kwargs:
        return lambda func: func
    else:
        return args[0]

try:
    import ray
    from ray import remote as ray_remote
    from ray import method as ray_method
except ImportError:
    ray = None
    ray_remote = dummy_decorator
    ray_method = dummy_decorator


class ParamServer:
    def __init__(self, model):
        self._handle = _ParamServer.remote(_get_model_params(model))

    def pull(self, model):
        _set_model_params(model, ray.get(self._handle.get_params.remote()))

    def push(self, model):
        params = {key: mem.weights for key, mem in _get_model_mems(model).items()}
        ray.get(self._handle.interpolate_params.remote(params))


def parallel_train(param_server, model, optimizer, Xs, Ys, batch_size, nproc):
    indices = list(range(len(Xs)))
    random.shuffle(indices)
    batches = list(minibatch(indices, size=batch_size))
    part_size = len(batches) // nproc
    losses = []
    for i in range(nproc):
        param_server.pull(model)
        chunk = batches[i * part_size : i * part_size + part_size]
        loss = _do_updates.remote(param_server, model, optimizer, Xs, Ys, chunk)
        losses.append(loss)
    return sum(map(ray.get, losses))
 

def _get_model_mems(model):
    queue = [model]
    output = {}
    for node in queue:
        if hasattr(node, "_mem"):
            output[node.id] = node._mem
        queue.extend(node._layers)
    return output

def _get_model_params(model):
    mems = _get_model_mems(model)
    return {key: mem.weights for key, mem in mems.items()}


def _set_model_params(model, params):
    mems = _get_model_mems(model)
    for key, param in params.items():
        mem = mems[key]
        mem.set(param)


@ray_remote(num_cpus=1)
def _do_updates(param_server, model, optimizer, Xs, Ys, batches):
    for key, mem in _get_model_mems(model).items():
        mem._grads_array = mem._grads_array.copy()
        mem._weights_array = mem._weights_array.copy()
    loss = 0.
    for batch in batches:
        X = Xs[batch]
        Y = Ys[batch]
        Yh, finish_update = model.begin_update(X, drop=0.)
        finish_update(Yh-Y, sgd=optimizer)
        loss += ((Y-Yh)**2).sum()
    param_server.push(model)
    return loss


@ray_remote
class _ParamServer:
    def __init__(self, params):
        self.params = {key: param for key, param in params.items()}

    @ray_method(num_return_vals=1)
    def get_params(self):
        return self.params

    @ray_method(num_return_vals=0)
    def set_params(self, params):
        self.params = params

    @ray_method(num_return_vals=1)
    def interpolate_params(self, params):
        new_params = {}
        for key, value in params.items():
            new_params[key] = 0.5 * self.params[key] + 0.5 * value
        self.params = new_params
        return True
