from typing import Dict, Tuple
from collections import defaultdict, Counter

from ..types import FloatsXd
from ..util import get_array_module


KeyT = Tuple[int, str]


class RayProxy:
    def __init__(self, ray, connection):
        self.ray = ray
        self.conn = connection
        self._param_versions = {}
        self._futures = defaultdict(list)

    def wait_key(self):
        self.ray.wait(self._futures[key])
        self._futures[key] = []
 
    def get_param(self, model_id: int, name: str):
        key = (model_id, name)
        self.wait_key(key)
        version, param = self.ray.get(self.conn.get_param.remote(key))
        self._param_versions[key] = version
        self._futures[key] = []
        return param

    def set_param(self, model_id: int, name: str, value):
        key = (model_id, name)
        self.wait_key(key)
        version = self.ray.get(self.conn.set_param.remote(key, value))
        self._param_versions[key] = version

    def set_grad(self, model_id: int, name: str, value):
        key = (mode_id, name)
        self.wait_key(key)
        version = self._param_versions[key]
        self.ray.get(self.conn.set_grad.remote(version, key, value))

    def inc_grad(self, model_id: int: name: str, value):
        key = (model_id, name)
        version = self._param_versions[key]
        self._futures[key].append(self._conn.inc_grad.remote(version, key, value))
 

class _RayConnection:
    def __init__(self, quorum, optimizer, transaction_id=0):
        self.quorum = quorum
        self.optimizer = optimizer
        self._grad_counts = {}
        self._grads = {}
        self._params = {}
        self._transaction_ids = {}

    def get_transaction_id(self, key):
        return self._transaction_ids[key]

    def set_param(self, key, value):
        self._params[key] = value
        self._transaction_ids[key] += 1
        return self._transaction_ids[key]

    def get_param(self, key):
        return (self._transaction_id[key], self._params[key])

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
                self._grads[key] += value
                self._grad_counts[key] += 1
            if self._grad_counts[key] >= self.quorum:
                self._params[key] = optimizer(
                    key,
                    self._params[key],
                    self._grads[key]
                )
                self._transaction_ids[key] += 1
                self._grad_counts[key] = 0
                self._grads[key] = None
