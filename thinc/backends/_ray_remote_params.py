from typing import Dict, Tuple
from collections import defaultdict, Counter

from ..types import FloatsXd
from ..util import get_array_module


KeyT = Tuple[int, str]


def make_key(model_id: int, name: str) -> int:


class RayHeadProxy:
    """Proxy for the 'head' worker that owns the optimizer and pushes
    parameter updates.
    """
    def __init__(self, connection, optimizer, quorum, *, ray=None):
        if ray is None:
            import ray
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self.optimizer = optimizer
        self.quorum = quorum
        self._param_versions = Counter()
        self._params = {}
        self._futures = defaultdict(list)

    def get_param(self, model_id: int, name: str):
        """Get a parameter from the connection."""
        key = (model_id, name)
        return self._params[key]

    def set_param(self, model_id: int, name: str, value):
        """Set a parameter to the connection."""
        key = (model_id, name)
        self._params[key] = value
        self._param_versions[key] += 1
        version = self._param_versions[key]
        self.conn.set_param.remote(
            version,
            model_id,
            name,
            self._encode_pointer(value)
        )

    def set_grad(self, model_id: int, name: str, value):
        """Set a gradient to the connection."""
        key = (mode_id, name)
        version = self._param_versions[key]
        self.conn.set_grad.remote(
            version,
            model_id,
            name,
            self._encode_pointer(value),
            1
        )

    def inc_grad(self, model_id: int, name: str, value):
        """Increment a gradient to the connection."""
        key = (model_id, name)
        param = self._params[key]
        version = self._param_versions[key]
        grad_count = self.ray.get(
            self.conn.get_grad_count.remote(
                version,
                model_id,
                name,
            )
        )
        if grad_count is None:
            raise ValueError("Gradient marked stale for head. Shouldn't happen?")
        else:
            if grad_count != 0:
                remote_grad = self._decode_pointer(
                    self.conn.get_grad.remote(
                        version,
                        model_id,
                        name,
                    )
                )
                if remote_grad is not None:
                    value += remote_grad

            if (grad_count + 1) >= self.quorum:
                param, _ = self.optimizer(key, param, value)
                self._params[key] = param
                self._param_versions[key] = version + 1
                self.conn.set_param.remote(
                    version+1,
                    model_id,
                    name,
                    self._encode_pointer(value)
                )
            else:
                self.conn.set_grad.remote(
                    version,
                    model_id,
                    name,
                    self._encode_pointer(value),
                    grad_count + 1
                )
 
    def step_schedules(self):
        self.optimizer.step_schedules()

    def _encode_pointer(self, value):
        return [self.ray.put(value)]
 
    def _decode_pointer(self, value):
        return self.ray.get(value)[0]


class RayChildProxy:
    def __init__(self, connection, *, ray=None):
        if ray is None:
            import ray
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self._param_versions = {}

    def get_param(self, model_id: int, name: str):
        """Get a parameter from the connection."""
        key = (model_id, name)
        version, param_id = self.ray.get(self.conn.get_param.remote(model_id, name))
        self._param_versions[key] = version
        return self._decode_pointer(param_id)

    def set_param(self, model_id: int, name: str, value):
        """Child proxies don't set parameters, so this is a noop."""
        pass

    def set_grad(self, model_id: int, name: str, value):
        """Child proxies don't set gradients, so this is a noop."""
        pass

    def inc_grad(self, model_id: int, name: str, value):
        """Increment a gradient to the connection."""
        key = (model_id, name)
        version = self._param_versions[key]
        grad_count = self.ray.get(
            self.conn.get_grad_count.remote(
                version,
                model_id,
                name
            )
        )
        if grad_count is None:
            return
        elif grad_count == 0:
            self.ray.get(
                self.conn.set_grad.remote(
                    version,
                    model_id,
                    name,
                    self._encode_pointer(value),
                    1
                )
            )
        else:
            remote_grad = self._decode_pointer(self.conn.get_grad.remote(
                version, model_id, name))
            if remote_grad is not None:
                value += remote_grad
            self.ray.get(
                self.conn.set_grad.remote(
                    version,
                    model_id,
                    name,
                    self._encode_pointer(value),
                    grad_count + 1
                )
            )

    def _encode_pointer(self, value):
        if value is None:
            return None
        else:
            return [self.ray.put(value)]
 
    def _decode_pointer(self, value):
        if value is None:
            return None
        else:
            return self.ray.get(value)[0]

    def _wait_key(self, key):
        """Await any futures for a given key."""
        self.ray.get(self._futures.get(key, []))
        self._futures[key] = []
 

class SharedParams:
    def __init__(self):
        self._grads = {}
        self._params = {}
        self._grad_counts = Counter()
        self._transaction_ids = Counter()
        self._progress = Counter()
 
    def inc_progress(self, worker_id):
        self._progress[worker_id] += 1
    
    def get_progress(self):
        return self._progress

    def get_total_progress(self):
        return sum(self._progress.values())

    def get_transaction_id(self, key):
        return self._transaction_ids[key]

    def get_param(self, model_id, name):
        key = (model_id, name)
        return (self._transaction_ids[key], self._params[key])

    def get_grad(self, version, model_id, name):
        key = (model_id, name)
        if self._transaction_ids.get(key) != version:
            return None
        else:
            return self._grads.get(key)

    def get_grad_count(self, version, model_id, name):
        key = (model_id, name)
        if self._transaction_ids.get(key) != version:
            return None
        elif key not in self._grad_counts:
            return 0
        else:
            return self._grad_counts[key]


    def set_param(self, version, model_id, name, value):
        key = (model_id, name)
        self._params[key] = value
        self._transaction_ids[key] = version
        # Discard gradients when we change version.
        self._grads[key] = None
        self._grad_counts[key] = 0

    def set_grad(self, tid, model_id, name, value, grad_count):
        key = (model_id, name)
        current_tid = self._transaction_ids.get(key)
        if tid != current_tid:
            # If we've moved past this version, discard the gradient.
            return None
        else:
            self._grads[key] = value
            self._grad_counts[key] = grad_count

    #def inc_grad(self, tid, model_id, name, value):
    #    key = (model_id, name)
    #    current_tid = self._transaction_ids[key]
    #    if tid != current_tid:
    #        # If we've moved past this version, discard the gradient.
    #        return None
    #    else:
    #        # Otherwise, increment the gradient
    #        if self._grads.get(key) is None:
    #            self._grads[key] = value
    #            self._grad_counts[key] = 1
    #        elif self._grad_counts[key] == 1:
    #            self._grads[key] = self._grads[key] + value
    #            self._grad_counts[key] += 1
    #        else:
    #            self._grads[key] += value
    #            self._grad_counts[key] += 1
    #        return self._grad_counts[key]

 
#class _RemoteOptimizer:
#    """Expose a thinc Optimizer instance as a remote task."""
#    def __init__(self, opt):
#        self.opt = opt
#
#    def call(self, key, params, grads):
#        params, grads = self.opt(key, params.copy(), grads.copy())
#        return params
#
#    def step_schedules(self):
#        return self.opt.step_schedules()
#
#
#class SharedOptimizer:
#    """Provide access to an optimizer for multiple workers. Designed to be
#    used as a ray remote actor, connected to a ParamServer via RayProxy.
#    """
#    def __init__(self, quorum, optimizer, ray=None, remote_optimizer=False):
#        if ray is None:
#            import ray
#        self.ray = ray
#        self.quorum = quorum
#        self.optimizer = optimizer
#        if remote_optimizer:
#            self.remote_optimizer = self.ray.remote(_RemoteOptimizer).remote(optimizer)
#        else:
#            self.remote_optimizer = None
#        self._grads = {}
#        self._params = {}
#        self._future_params = {}
#        self._grad_counts = Counter()
#        self._transaction_ids = Counter()
#        self._progress = Counter()
#
#    def get_quorum(self):
#        return self.quorum
#
#    def inc_progress(self, worker_id):
#        self._progress[worker_id] += 1
#    
#    def get_progress(self):
#        return self._progress
#
#    def get_total_progress(self):
#        return sum(self._progress.values())
#
#    def step_schedules(self):
#        if self.optimizer is not None:
#            self.optimizer.step_schedules()
#        if self.remote_optimizer is not None:
#            self.ray.get(self.remote_optimizer.step_schedules.remote())
#
#    def get_transaction_id(self, key):
#        return self._transaction_ids[key]
#
#    def get_param(self, model_id, name):
#        key = (model_id, name)
#        if key in self._future_params:
#            self._params[key] = self.ray.get(self._future_params.pop(key))
#        return (self._transaction_ids[key], self._params[key])
#
#    def set_param(self, model_id, name, value):
#        key = (model_id, name)
#        if key in self._future_params:
#            self._params[key] = self.ray.get(self._future_params.pop(key))
#        self._params[key] = value
#        self._transaction_ids[key] += 1
#        # Discard gradients when we change version.
#        self._grads[key] = None
#        self._grad_counts[key] = 0
#        return self._transaction_ids[key]
#
#    def set_grad(self, tid, model_id, name, value):
#        key = (model_id, name)
#        current_tid = self._transaction_ids[key]
#        if tid != current_tid:
#            # If we've moved past this version, discard the gradient.
#            return None
#        elif key in self._future_params:
#            return None
#        else:
#            self._grads[key] = value
#            self._grad_counts[key] = 1
#            self._update_if_quorum(key)
#
#    def inc_grad(self, tid, model_id, name, value):
#        key = (model_id, name)
#        current_tid = self._transaction_ids[key]
#        if tid != current_tid:
#            # If we've moved past this version, discard the gradient.
#            return None
#        elif key in self._future_params:
#            return None
#        else:
#            # Otherwise, increment the gradient
#            if self._grads.get(key) is None:
#                self._grads[key] = value
#                self._grad_counts[key] = 1
#            else:
#                self._grads[key] = self._grads[key] + value
#                self._grad_counts[key] += 1
#            self._update_if_quorum(key)
#
#    def _update_if_quorum(self, key):
#        if self._grad_counts[key] >= self.quorum:
#            if self.remote_optimizer is not None:
#                self._future_params[key] = self.remote_optimizer.call.remote(
#                    key,
#                    self._params[key],
#                    self._grads[key]
#                )
#            else:
#                params, grads = self.optimizer(
#                    key,
#                    self._params[key].copy(),
#                    self._grads[key].copy()
#                )
#                self._params[key] = params
# 
#            self._transaction_ids[key] += 1
#            self._grad_counts[key] = 0
#            self._grads[key] = None
