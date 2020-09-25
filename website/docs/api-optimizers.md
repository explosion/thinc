---
title: Optimizers
next: /docs/api-initializers
---

An optimizer essentially performs stochastic gradient descent. It takes
one-dimensional arrays for the weights and their gradients, along with an
optional identifier key. The optimizer is expected to update the weights and
zero the gradients in place. The optimizers are registered in the
[function registry](/docs/api-config#registry) and can also be used via Thinc's
[config mechanism](/docs/usage-config).

## Optimizer functions

### SGD {#sgd tag="function"}

If a hyperparameter specifies a schedule as a list or generator, its value will
be replaced with the next item on each call to
[`Optimizer.step_schedules`](#step-schedules). Once the schedule is exhausted,
its last value will be used.

<grid>

```python
### Example {small="true"}
from thinc.api import SGD

optimizer = SGD(
    learn_rate=0.001,
    L2=1e-6,
    grad_clip=1.0
)
```

```ini
### config.cfg {small="true"}
[optimizer]
@optimizers = SGD.v1
learn_rate = 0.001
L2 = 1e-6
L2_is_weight_decay = true
grad_clip = 1.0
use_averages = true
```

</grid>

| Argument             | Type                                          | Description                                                                                        |
| -------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `learn_rate`         | <tt>Union[float, List[float], Generator]</tt> | The initial learning rate.                                                                         |
| _keyword-only_       |                                               |                                                                                                    |
| `L2`                 | <tt>Union[float, List[float], Generator]</tt> | The L2 regularization term.                                                                        |
| `grad_clip`          | <tt>Union[float, List[float], Generator]</tt> | Gradient clipping.                                                                                 |
| `use_averages`       | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                |
| `L2_is_weight_decay` | <tt>bool</tt>                                 | Whether to interpret the L2 parameter as a weight decay term, in the style of the AdamW optimizer. |
| `ops`                | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                      |

### Adam {#adam tag="function"}

Function to create an Adam optimizer. Returns an instance of
[`Optimizer`](#optimizer). If a hyperparameter specifies a schedule as a list or
generator, its value will be replaced with the next item on each call to
[`Optimizer.step_schedules`](#step-schedules). Once the schedule is exhausted,
its last value will be used.

<grid>

```python
### Example {small="true"}
from thinc.api import Adam

optimizer = Adam(
    learn_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-08,
    L2=1e-6,
    grad_clip=1.0,
    use_averages=True,
    L2_is_weight_decay=True
)
```

```ini
### config.cfg {small="true"}
[optimizer]
@optimizers = Adam.v1
learn_rate = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-08
L2 = 1e-6
L2_is_weight_decay = true
grad_clip = 1.0
use_averages = true
```

</grid>

| Argument             | Type                                          | Description                                                                                        |
| -------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `learn_rate`         | <tt>Union[float, List[float], Generator]</tt> | The initial learning rate.                                                                         |
| _keyword-only_       |                                               |                                                                                                    |
| `L2`                 | <tt>Union[float, List[float], Generator]</tt> | The L2 regularization term.                                                                        |
| `beta1`              | <tt>Union[float, List[float], Generator]</tt> | First-order momentum.                                                                              |
| `beta2`              | <tt>Union[float, List[float], Generator]</tt> | Second-order momentum.                                                                             |
| `eps`                | <tt>Union[float, List[float], Generator]</tt> | Epsilon term for Adam etc.                                                                         |
| `grad_clip`          | <tt>Union[float, List[float], Generator]</tt> | Gradient clipping.                                                                                 |
| `use_averages`       | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                |
| `L2_is_weight_decay` | <tt>bool</tt>                                 | Whether to interpret the L2 parameter as a weight decay term, in the style of the AdamW optimizer. |
| `ops`                | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                      |

### RAdam {#radam tag="function"}

Function to create an RAdam optimizer. Returns an instance of
[`Optimizer`](#optimizer). If a hyperparameter specifies a schedule as a list or
generator, its value will be replaced with the next item on each call to
[`Optimizer.step_schedules`](#step-schedules). Once the schedule is exhausted,
its last value will be used.

<grid>

```python
### Example {small="true"}
from thinc.api import RAdam

optimizer = RAdam(
    learn_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-08,
    weight_decay=1e-6,
    grad_clip=1.0,
    use_averages=True,
)
```

```ini
### config.cfg {small="true"}
[optimizer]
@optimizers = RAdam.v1
learn_rate = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-08
weight_decay = 1e-6
grad_clip = 1.0
use_averages = true
```

</grid>

| Argument       | Type                                          | Description                                                   |
| -------------- | --------------------------------------------- | ------------------------------------------------------------- |
| `learn_rate`   | <tt>Union[float, List[float], Generator]</tt> | The initial learning rate.                                    |
| _keyword-only_ |                                               |                                                               |
| `beta1`        | <tt>Union[float, List[float], Generator]</tt> | First-order momentum.                                         |
| `beta2`        | <tt>Union[float, List[float], Generator]</tt> | Second-order momentum.                                        |
| `eps`          | <tt>Union[float, List[float], Generator]</tt> | Epsilon term for Adam etc.                                    |
| `weight_decay` | <tt>Union[float, List[float], Generator]</tt> | Weight decay term.                                            |
| `grad_clip`    | <tt>Union[float, List[float], Generator]</tt> | Gradient clipping.                                            |
| `use_averages` | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.           |
| `ops`          | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend. |

---

## Optimizer {tag="class"}

Do various flavors of stochastic gradient descent, with first and second order
momentum. Currently support "vanilla" SGD, Adam, and RAdam.

### Optimizer.\_\_init\_\_ {#init tag="method"}

Initialize an optimizer. If a hyperparameter specifies a schedule as a list or
generator, its value will be replaced with the next item on each call to
[`Optimizer.step_schedules`](#step-schedules). Once the schedule is exhausted,
its last value will be used.

```python
### Example
from thinc.api import Optimizer

optimizer = Optimizer(learn_rate=0.001, L2=1e-6, grad_clip=1.0)
```

| Argument             | Type                                          | Description                                                                                        |
| -------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `learn_rate`         | <tt>Union[float, List[float], Generator]</tt> | The initial learning rate.                                                                         |
| _keyword-only_       |                                               |                                                                                                    |
| `L2`                 | <tt>Union[float, List[float], Generator]</tt> | The L2 regularization term.                                                                        |
| `beta1`              | <tt>Union[float, List[float], Generator]</tt> | First-order momentum.                                                                              |
| `beta2`              | <tt>Union[float, List[float], Generator]</tt> | Second-order momentum.                                                                             |
| `eps`                | <tt>Union[float, List[float], Generator]</tt> | Epsilon term for Adam etc.                                                                         |
| `grad_clip`          | <tt>Union[float, List[float], Generator]</tt> | Gradient clipping.                                                                                 |
| `use_averages`       | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                |
| `use_radam`          | <tt>bool</tt>                                 | Whether to use the RAdam optimizer.                                                                |
| `L2_is_weight_decay` | <tt>bool</tt>                                 | Whether to interpret the L2 parameter as a weight decay term, in the style of the AdamW optimizer. |
| `ops`                | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                      |

### Optimizer.\_\_call\_\_ {#call tag="method"}

Call the optimizer function, updating parameters using the current parameter
gradients. The `key` is the identifier for the parameter, usually the node ID
and parameter name.

| Argument       | Type                     | Description                                   |
| -------------- | ------------------------ | --------------------------------------------- |
| `key`          | <tt>Tuple[int, str]</tt> | The parameter identifier.                     |
| `weights`      | <tt>FloatsXd</tt>        | The model's current weights.                  |
| `gradient`     | <tt>FloatsXd</tt>        | The model's current gradient.                 |
| _keyword-only_ |                          |                                               |
| `lr_scale`     | <tt>float</tt>           | Rescale the learning rate. Defaults to `1.0`. |

### Optimizer.step_schedules {#step_schedules tag="method"}

Replace the the named hyperparameters with the next item from the schedules
iterator, if available. Once the schedule is exhausted, its last value will be
used.

```python
### Example
from thinc.api import Optimizer, decaying

optimizer = Optimizer(learn_rate=decaying(0.001, 1e-4), grad_clip=1.0)
assert optimizer.learn_rate == 0.001
optimizer.step_schedules()
assert optimizer.learn_rate == 0.000999900009999  # using a schedule
assert optimizer.grad_clip == 1.0                 # not using a schedule
```

### Optimizer.to_gpu {#to_gpu tag="method"}

Transfer the optimizer to a given GPU device.

```python
### Example
optimizer.to_gpu()
```

### Optimizer.to_cpu {#to_cpu tag="method"}

Copy the optimizer to CPU.

```python
### Example
optimizer.to_cpu()
```

### Optimizer.to_gpu {#to_gpu tag="method"}

Transfer the optimizer to a given GPU device.

```python
### Example
optimizer.to_gpu()
```

### Optimizer.to_cpu {#to_cpu tag="method"}

Copy the optimizer to CPU.

```python
### Example
optimizer.to_cpu()
```
