from typing import Dict, Tuple
from collections import defaultdict, Counter

from ..types import FloatsXd
from ..util import get_array_module


KeyT = Tuple[int, str]


class RayProxy:
    def __init__(self, connection, *, ray=None):
        if ray is None:
            import ray
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        print("Create proxy", type(self.conn))
        self._param_versions = {}
        self._futures = defaultdict(list)

    def wait_key(self, key):
        """Await any futures for a given key."""
        self.ray.wait(self._futures[key], len(self._futures[key]))
        self._futures[key] = []
 
    def get_param(self, model_id: int, name: str):
        """Get a parameter from the connection."""
        key = (model_id, name)
        self.wait_key(key)
        version, param = self.ray.get(self.conn.get_param.remote(key))
        self._param_versions[key] = version
        self._futures[key] = []
        return param

    def set_param(self, model_id: int, name: str, value):
        """Set a parameter to the connection."""
        key = (model_id, name)
        self.wait_key(key)
        version = self.ray.get(self.conn.set_param.remote(key, value))
        self._param_versions[key] = version

    def set_grad(self, model_id: int, name: str, value):
        """Set a gradient to the connection."""
        key = (mode_id, name)
        self.wait_key(key)
        version = self._param_versions[key]
        self.ray.get(self.conn.set_grad.remote(version, key, value))

    def inc_grad(self, model_id: int, name: str, value):
        """Increment a gradient to the connection."""
        key = (model_id, name)
        version = self._param_versions[key]
        self._futures[key].append(self.conn.inc_grad.remote(version, key, value))
 

class SharedOptimizer:
    """Provide access to an optimizer for multiple workers. Designed to be
    used as a ray remote actor, connected to a ParamServer via RayProxy.
    """
    def __init__(self, quorum, optimizer):
        self.quorum = quorum
        self.optimizer = optimizer
        self._grads = {}
        self._params = {}
        self._grad_counts = Counter()
        self._transaction_ids = Counter()

    def get_transaction_id(self, key):
        return self._transaction_ids[key]

    def get_param(self, key):
        return (self._transaction_ids[key], self._params[key])

    def set_param(self, key, value):
        self._params[key] = value
        self._transaction_ids[key] += 1
        # Discard gradients when we change version.
        self._grads[key] = None
        self._grad_counts[key] = 0
        return self._transaction_ids[key]

    def set_grad(self, tid, key, value):
        current_tid = self._transaction_ids[key]
        if tid != current_tid:
            # If we've moved past this version, discard the gradient.
            return None
        else:
            self._grads[key] = value
            self._grad_counts[key] = 1
            self._update_if_quorum(key)

    def inc_grad(self, tid, key, value):
        current_tid = self._transaction_ids[key]
        if tid != current_tid:
            # If we've moved past this version, discard the gradient.
            return None
        else:
            # Otherwise, increment the gradient
            if self._grads.get(key) is None:
                self._grads[key] = value
                self._grad_counts[key] = 1
            else:
                self._grads[key] = self._grads[key] + value
                self._grad_counts[key] += 1
            self._update_if_quorum(key)

    def _update_if_quorum(self, key):
        if self._grad_counts[key] >= self.quorum:
            self._params[key]
            params, grads = self.optimizer(
                key,
                self._params[key],
                self._grads[key]
            )
            self._params[key] = params
            self._transaction_ids[key] += 1
            self._grad_counts[key] = 0
            self._grads[key] = None
