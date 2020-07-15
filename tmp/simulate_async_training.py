from typing import List
from dataclasses import dataclass
import random
import typer
import numpy.random
from collections import Counter


@dataclass
class Event:
    worker: int
    time: int
    duration: int
    msg: str

    @property
    def end_time(self):
        return self.time + self.duration

    def __str__(self):
        return f"{self.time}\t{self.worker}\t{self.msg}"


class Layer:
    def __init__(self, i):
        self.i = i

    def forward(self, batch):
        duration = numpy.random.normal(0.5, scale=0.2)
        msg = f"forward_{self.i}"
        return (msg, duration)
    
    def backward(self, batch):
        duration = numpy.random.normal(0.6, scale=0.3)
        msg = f"backward_{self.i}"
        return (msg, duration)


class Worker:
    def __init__(self, id, network, batches, time=0):
        self.id = id
        self.time = time
        self.total_blocked_time = 0
        self.network = network
        self.batches = list(batches)
        self._batch_i = -1
        self._refresh_state()
        self.is_blocked = False

    def _refresh_state(self):
        self._batch_i += 1
        if self._batch_i >= len(self.batches):
            random.shuffle(self.batches)
            self._batch_i = 0
        self._state = []
        for layer in self.network:
            self._state.append(layer.forward)
        for layer in reversed(self.network):
            self._state.append(layer.backward)

    def next_event(self):
        if not self._state:
            self._refresh_state()
        start_time = self.time
        msg, duration = self._state.pop(0)(self.batches[self._batch_i])
        self.time += duration
        return Event(self.id, start_time, duration, msg)

    def block(self):
        self.is_blocked = True

    def unblock(self, time):
        self.total_blocked_time += time - self.time
        self.time = time
        self.is_blocked = False


def get_batches(n_batches: int, batch_size: int, mean_seq_length: int) -> List[List[int]]:
    batches: List[List[int]] = []
    for _ in range(n_batches):
        batches.append([
            max(1, int(numpy.random.normal(scale=mean_seq_length)))
            for _ in range(batch_size)
        ])
    return batches


def get_network(n_layers: int) -> List[Layer]:
    return [Layer(i) for i in range(n_layers)]


def get_workers(network, batches, n_workers):
    workers = []
    start = 0
    shard_size = len(batches) // n_workers
    for i in range(n_workers):
        workers.append(Worker(i, network, batches[start : start + shard_size]))
        start += shard_size
    return workers


def simulate_synchronous(n, network, workers):
    """Simulate synchronous training, where we block the workers at the end
    of the backward pass until all workers have completed. Measure the percentage
    of total worker-time wasted, and return a sorted series of events.
    """
    events = []
    while len(events) < n:
        for worker in workers:
            while not worker.is_blocked:
                event = worker.next_event()
                events.append(event)
                if event.msg == "backward_0":
                    worker.block()
                    if all(w.is_blocked for w in workers):
                        time = max(w.time for w in workers)
                        for w in workers:
                            w.unblock(time)
                        break
    events.sort(key=lambda ev: ev.time)
    blocked_time = sum(worker.total_blocked_time for worker in workers)
    total_time = sum(event.duration for event in events) + blocked_time
    return blocked_time / total_time, events


def simulate_lock_free(n, network, workers, quorum):
    """Simulate "lock free" training, where we let the workers continue, but
    discard stale gradients.
    """
    events = []
    while len(events) < n:
        for worker in workers:
            event = worker.next_event()
            events.append(event)
    events.sort(key=lambda ev: ev.time)
    n_grads = Counter()
    wasted_time = 0
    for event in events:
        if event.msg.startswith("backward"):
            n_grads[event.msg] += 1
            if n_grads[event.msg] > quorum:
                wasted_time += event.duration
        elif event.msg.startswith("forward"):
            if n_grads[event.msg.replace("forward", "backward")] >= quorum:
                n_grads[event.msg.replace("forward", "backward")] = 0
    total_time = sum(event.duration for event in events)
    return wasted_time / total_time, events


def main(
    n_layers: int=3,
    n_batches: int=100,
    batch_size: int=8,
    mean_seq_length: int=100,
    n_workers: int=2,
    quorum: int=4
):
    random.seed(0)
    numpy.random.seed(0)
    batches = get_batches(n_batches, batch_size, mean_seq_length)
    network = get_network(n_layers)
    workers = get_workers(network, batches, n_workers)
    print("Synchronous")
    waste, events = simulate_synchronous(500, network, workers)
    print("Wasted", waste)
    print("Lock free")
    waste, events = simulate_lock_free(
        500,
        network,
        get_workers(network, batches, n_workers),
        quorum
    )
    print("Wasted", waste)


if __name__ == "__main__":
    typer.run(main)
