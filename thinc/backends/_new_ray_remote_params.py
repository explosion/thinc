from typing import Dict, Tuple, Any
from timeit import default_timer as timer
import threading
from dataclasses import dataclass
from collections import Counter, defaultdict
import time
import threading
from collections import UserDict
from ..types import FloatsXd


KeyT = Tuple[int, str]


def make_key(model_id: int, name: str) -> Tuple[int, str]:
    return (model_id, name)

class Timer:
    def __init__(self, state):
        self.state = state
        self.sum = 0
        self.n = 0

    def __enter__(self):
        self.start = time.time()
        self.n += 1

    def __exit__(self, *args):
        interval = time.time() - self.start
        self.sum += interval
        print(f"{self.state}: {self.sum / self.n:0.4f}")


class ManyTimer:
    def __init__(self):
        self.timers = {}

    def __call__(self, key):
        if key not in self.timers:
            self.timers[key] = Timer(key)
        return self.timers[key]

def thread_function(next_params, ray_, conn, poll):
    _last_update = 0
    while True:
        time.sleep(poll)
        updates = ray_.get(
            conn.get_updated_params.remote(_last_update)
        )
        new_time = timer()
        _last_update = new_time
        next_params.update(updates)

class RayProxy:
    """Proxy for the 'head' worker that owns the optimizer and pushes
    parameter updates.
    """
    ray: Any
    conn: Any
    _params: Dict
    _next_params: Dict
    _versions: Dict

    def __init__(self, connection, *, ray=None, use_thread=False, poll_every=0.1):
        if ray is None:
            import ray # type: ignore
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self._poll_every = poll_every
        self._last_update = 0
        self._next_params = {}
        self._params = {}
        self._versions = {}
        self._grad_futures = defaultdict(list)
        self.timers = ManyTimer()
        self.use_thread = use_thread
        if self.use_thread:
            args = (
                self._next_params,
                self.ray,
                self.conn,
                self._poll_every
            )
            self.thread = threading.Thread(
                target=thread_function,
                args=args,
                daemon=True
            )

    def set_param(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a parameter to the connection."""
        key = make_key(model_id, name)
        self._params[key] = value
        self._versions[key] = self.ray.get(
            self.conn.set_param.remote(
                key,
                value
            )
        )

    def get_param(self, model_id: int, name: str) -> FloatsXd:
        """Get a parameter from the connection."""
        key = make_key(model_id, name)
        if not self.use_thread and (timer() - self._last_update) >= self._poll_every:
            self._refresh_nexts()
        self._maybe_update_param(key)
        return self._params[key]

    def set_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a gradient to the connection."""
        key = make_key(model_id, name)
        self._grad_futures[key].append(
            self.conn.set_grad.remote(
                key,
                self._versions[key],
                value
            )
        )

    def inc_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Increment a gradient to the connection."""
        key = make_key(model_id, name)
        self._grad_futures[key].append(
            self.conn.inc_grad.remote(
                key,
                self._versions[key],
                value
            )
        )

    def _refresh_nexts(self):
        self._await_grads()
        now_time = timer()
        self._next_params.update(
            self.ray.get(
                self.conn.get_updated_params.remote(self._last_update)
            )
        )
        self._last_update = now_time

    def _await_grads(self):
        futures = []
        for g in self._grad_futures.values():
            futures.extend(g)
        self.ray.get(futures)
        self._grad_futures = defaultdict(list)
        
    def _maybe_update_param(self, key):
        if key in self._next_params:
            self._versions[key], self._params[key] = self._next_params.pop(key)
        if key in self._grad_futures:
            self.ray.get(self._grad_futures.pop(key))
            maybe_param = self.ray.get(
                self.conn.get_param_if_updated.remote(key, self._versions[key])
            )
            if maybe_param is not None:
                self._versions[key], self._params[key] = maybe_param
                return True
        return False

    def _sync_params(self):
        self._await_grads()
        self._params = {}
        self._versions = {}
        self._next_params = {}
        params = self.ray.get(self.conn.get_updated_params.remote(0))
        for key, (version, param) in params.items():
            self._params[key] = param
            self._versions[key] = version


ObjectID = int

@dataclass
class ParamData:
    key: Tuple[int, str]
    version: int
    timestamp: Any
    value: FloatsXd
    grads: FloatsXd
    grad_count: int


class SharedOptimizer:
    """Provide access to an optimizer for multiple workers. Designed to be
    used as a ray remote actor, connected to a ParamServer via RayProxy.
    """
    def __init__(self, optimizer, quorum, ray=None):
        if ray is None:
            import ray
        self.ray = ray
        self.quorum = quorum
        self.optimizer = optimizer
        self._params = {}
        self._progress = Counter()
        self._n_kept = 0
        self._n_dropped = 0
        self._n_updates = 0
        self.write_lock = threading.Lock()

    def get_quorum(self):
        return self.quorum

    def inc_progress(self, worker_id):
        self._progress[worker_id] += 1

    def get_progress(self):
        return self._progress

    def get_total_progress(self):
        return sum(self._progress.values())

    def step_schedules(self):
        self.optimizer.step_schedules()

    def get_transaction_id(self, key):
        return self._params[key].version

    def get_param(self, key):
        return (self._params[key].version, self._params[key].value)

    def get_percent_dropped(self):
        total = self._n_dropped + self._n_kept
        if total == 0:
            return total
        else:
            return self._n_dropped / total

    def get_param_if_updated(self, key, version):
        if key not in self._params:
            raise KeyError("wat")
        elif self._params[key].version == version:
            return None
        else:
            return (self._params[key].version, self._params[key].value)

    def get_updated_params(self, since: float) -> Dict:
        """Return a dict with params that have changed since a given timestamp.
        """
        updates = {}
        for key, p in self._params.items():
            if p.timestamp >= since:
                updates[key] = (p.version, p.value)
        return updates

    def set_param(self, key, value):
        with self.write_lock:
            if key in self._params:
                version = self._params[key].version + 1
            else:
                version = 0
            self._params[key] = ParamData(
                key=key,
                value=value,
                version=version,
                grads=None,
                grad_count=0,
                timestamp=timer()
            )
            return self._params[key].version

    def set_grad(self, tid, key, value):
        with self.write_lock:
            if key not in self._params:
                return None
            elif tid != self._params[key].version:
                # If we've moved past this version, discard the gradient.
                return None
            else:
                self._params[key].grads = value.copy()
                self._params[key].grad_count = 1
                self._update_if_quorum(key)

    def inc_grad(self, key, tid, value):
        with self.write_lock:
            if key not in self._params:
                return None
            elif tid != self._params[key].version:
                self._n_dropped += 1
                return None
            elif self._params[key].grads is None:
                self._n_kept += 1
                self._params[key].grads = value.copy()
                self._params[key].grad_count = 1
                self._update_if_quorum(key)
            else:
                self._n_kept += 1
                self._params[key].grads += value
                self._params[key].grad_count += 1
                self._update_if_quorum(key)

    def _update_if_quorum(self, key):
        if key not in self._params:
            return
        if self._params[key].grad_count >= self.quorum:
            # The optimizer call changes internal state, so we need to worry
            # about concurrency on it.
            params, _ = self.optimizer(
                key,
                self._params[key].value.copy(),
                self._params[key].grads
            )
            new_param = ParamData(
                key=key,
                value=params,
                version=self._params[key].version + 1,
                grads=None,
                grad_count=0,
                timestamp=timer()
            )
            self._params[key] = new_param
            self._n_updates += 1
