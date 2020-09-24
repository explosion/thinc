---
title: Training Models
next: /docs/usage-frameworks
---

## Basic training loop {#training-loop}

```python
from thinc.api import minibatch, Adam

optimizer = Adam(0.001)
indices = model.ops.xp.arange(train_X.shape[0], dtype="i")
for i in range(n_iter):
    model.ops.xp.random.shuffle(indices)
    for idx_batch in minibatch(indices):
        Yh, backprop = model.begin_update(train_X[idx_batch])
        backprop(Yh - train_Y[idx_batch])
        model.finish_update(optimizer)
```

The `backprop` callback increments the gradients of the parameters, but does not
change the parameters themselves. This makes gradient accumulation trivial: just
don't call [`model.finish_update`](/docs/api-model#finish_update), and the
gradients will accumulate. Gradient accumulation is especially important for
transformer models, which often work best with larger batch sizes than can
easily be fit on a GPU.

## Distributed training {#distributed}

We expect to recommend [Ray](https://ray.io/) for distributed training. Ray
offers a clean and simple API that fits well with Thinc's model design. While
full support is still under development, you can find a draft example here:

```python
https://github.com/explosion/thinc/blob/fec03fc448670a1b57baf8f8b825ddaef88b57f3/examples/scripts/ray_parallel.py
```

## Setting learning rate schedules {#schedules}

A common trick for stochastic gradient descent is to **vary the learning rate or
other hyperparameters** over the course of training. Since there are many
possible ways to vary the learning rate, Thinc lets you implement hyperparameter
schedules as simple generator functions. Thinc also provides a number of popular
schedules built-in.

You can use schedules directly, by calling `next()` on the schedule and using it
to update hyperparameters in your training loop. Since schedules are
particularly common for optimization settings, the
[`Optimizer`](/docs/api-optimizer) object also accepts a dictionary mapping
attribute names to iterables. When you call
[`Optimizer.step_schedules`](/docs/api-optimizer#step_schedules), the optimizer
will draw the next value from the generators in its `schedules` dictionary, and
use them to change the given attributes. For instance, here's how to create an
instance of the `Adam` optimizer with a custom learning rate schedule:

```python
from thinc.api import Adam

def my_schedule():
    values = [0.001, 0.01, 0.1]
    while True:
        for value in values:
            yield value
        for value in reverse(values):
            yield value

optimizer = Adam(1.0, schedules={"learn_rate": my_schedule()})
assert optimizer.learn_rate == 1.0
optimizer.step_schedules()
assert optimizer.learn_rate == 0.001
optimizer.step_schedules()
assert optimizer.learn_rate == 0.01
```

You'll often want to describe your optimization schedules in your configuration
file. That's also very easy: check out the
[documentation on config files](/docs/usage-config) for an example.
