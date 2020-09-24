---
title: Optimizers
next: /docs/api-initializers
---

An optimizer essentially performs stochastic gradient descent. It takes
1-dimensional arrays for the weights and their gradients, along with an optional
identifier key. The optimizer is expected to update the weights and zero the
gradients in place. The optimizers are registered in the
[function registry](/docs/api-config#registry) and can also be used via Thinc's
[config mechanism](/docs/usage-config).

## Optimizer functions

### SGD {#sgd tag="function"}

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

| Argument             | Type                                          | Description                                                                                                                                                                           |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learn_rate`         | <tt>float</tt>                                | The initial learning rate.                                                                                                                                                            |
| _keyword-only_       |                                               |                                                                                                                                                                                       |
| `ops`                | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                                                                                                         |
| `L2`                 | <tt>float</tt>                                | The L2 regularization term.                                                                                                                                                           |
| `grad_clip`          | <tt>float</tt>                                | Gradient clipping.                                                                                                                                                                    |
| `use_averages`       | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                                                                                                   |
| `L2_is_weight_decay` | <tt>bool</tt>                                 | Whether to interpret the L2 parameter as a weight decay term, in the style of the AdamW optimizer.                                                                                    |
| `schedules`          | <tt>Optional[Dict[str, Iterator[float]]]</tt> | Dictionary mapping hyperparameter names to value iterators. On each call to `Optimizer.step_schedules`, the named hyperparameters are replaced with the next item from the generator. |

### Adam {#adam tag="function"}

Function to create an Adam optimizer. Returns an instance of
[`Optimizer`](#optimizer).

<grid>

```python
### Example {small="true"}
from thinc.api import Adam

optimizer = Adam(
    learn_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-08
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
lookahead_k = 0
lookeahead_alpha = 0.5
```

</grid>

| Argument             | Type                                          | Description                                                                                                                                                                           |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learn_rate`         | <tt>float</tt>                                | The initial learning rate.                                                                                                                                                            |
| _keyword-only_       |                                               |                                                                                                                                                                                       |
| `ops`                | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                                                                                                         |
| `L2`                 | <tt>float</tt>                                | The L2 regularization term.                                                                                                                                                           |
| `beta1`              | <tt>float</tt>                                | First-order momentum.                                                                                                                                                                 |
| `beta2`              | <tt>float</tt>                                | Second-order momentum.                                                                                                                                                                |
| `eps`                | <tt>float</tt>                                | Epsilon term for Adam etc.                                                                                                                                                            |
| `grad_clip`          | <tt>float</tt>                                | Gradient clipping.                                                                                                                                                                    |
| `lookahead_k`        | <tt>int</tt>                                  | K parameter for lookahead.                                                                                                                                                            |
| `lookahead_alpha`    | <tt>float</tt>                                | Alpha parameter for lookahead.                                                                                                                                                        |
| `use_averages`       | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                                                                                                   |
| `L2_is_weight_decay` | <tt>bool</tt>                                 | Whether to interpret the L2 parameter as a weight decay term, in the style of the AdamW optimizer.                                                                                    |
| `schedules`          | <tt>Optional[Dict[str, Iterator[float]]]</tt> | Dictionary mapping hyperparameter names to value iterators. On each call to `Optimizer.step_schedules`, the named hyperparameters are replaced with the next item from the generator. |

### RAdam {#radam tag="function"}

Function to create an RAdam optimizer. Returns an instance of
[`Optimizer`](#optimizer).

<grid>

```python
### Example {small="true"}
from thinc.api import RAdam

optimizer = RAdam(
    learn_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-08
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
lookahead_k = 0
lookeahead_alpha = 0.5
```

</grid>

| Argument          | Type                                          | Description                                                                                                                                                                           |
| ----------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learn_rate`      | <tt>float</tt>                                | The initial learning rate.                                                                                                                                                            |
| _keyword-only_    |                                               |                                                                                                                                                                                       |
| `ops`             | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                                                                                                         |
| `beta1`           | <tt>float</tt>                                | First-order momentum.                                                                                                                                                                 |
| `beta2`           | <tt>float</tt>                                | Second-order momentum.                                                                                                                                                                |
| `eps`             | <tt>float</tt>                                | Epsilon term for Adam etc.                                                                                                                                                            |
| `weight_decay`    | <tt>float</tt>                                | Weight decay term.                                                                                                                                                                    |
| `grad_clip`       | <tt>float</tt>                                | Gradient clipping.                                                                                                                                                                    |
| `lookahead_k`     | <tt>int</tt>                                  | K parameter for lookahead.                                                                                                                                                            |
| `lookahead_alpha` | <tt>float</tt>                                | Alpha parameter for lookahead.                                                                                                                                                        |
| `use_averages`    | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                                                                                                   |
| `schedules`       | <tt>Optional[Dict[str, Iterator[float]]]</tt> | Dictionary mapping hyperparameter names to value iterators. On each call to `Optimizer.step_schedules`, the named hyperparameters are replaced with the next item from the generator. |

---

## Optimizer {tag="class"}

Do various flavors of stochastic gradient descent, with first and second order
momentum. Currently support "vanilla" SGD, Adam, and RAdam.

### Optimizer.\_\_init\_\_ {#init tag="method"}

Initialize an optimizer.

```python
### Example
from thinc.api import Optimizer

optimizer = Optimizer(learn_rate=0.001, L2=1e-6, grad_clip=1.0)
```

| Argument             | Type                                          | Description                                                                                                                                                                           |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learn_rate`         | <tt>float</tt>                                | The initial learning rate.                                                                                                                                                            |
| _keyword-only_       |                                               |                                                                                                                                                                                       |
| `ops`                | <tt>Optional[Ops]</tt>                        | A backend object. Defaults to the currently selected backend.                                                                                                                         |
| `L2`                 | <tt>float</tt>                                | The L2 regularization term.                                                                                                                                                           |
| `beta1`              | <tt>float</tt>                                | First-order momentum.                                                                                                                                                                 |
| `beta2`              | <tt>float</tt>                                | Second-order momentum.                                                                                                                                                                |
| `eps`                | <tt>float</tt>                                | Epsilon term for Adam etc.                                                                                                                                                            |
| `grad_clip`          | <tt>float</tt>                                | Gradient clipping.                                                                                                                                                                    |
| `lookahead_k`        | <tt>int</tt>                                  | K parameter for lookahead.                                                                                                                                                            |
| `lookahead_alpha`    | <tt>float</tt>                                | Alpha parameter for lookahead.                                                                                                                                                        |
| `use_averages`       | <tt>bool</tt>                                 | Whether to track moving averages of the parameters.                                                                                                                                   |
| `use_radam`          | <tt>bool</tt>                                 | Whether to use the RAdam optimizer.                                                                                                                                                   |
| `L2_is_weight_decay` | <tt>bool</tt>                                 | Whether to interpret the L2 parameter as a weight decay term, in the style of the AdamW optimizer.                                                                                    |
| `schedules`          | <tt>Optional[Dict[str, Iterable[float]]]</tt> | Dictionary mapping hyperparameter names to value iterables. On each call to `Optimizer.step_schedules`, the named hyperparameters are replaced with the next item from the generator. |

### Optimizer.\_\_call\_\_ {#call tag="method"}

Call the optimizer function, updating parameters using the current parameter
gradients.

| Argument   | Type           | Description                   |
| ---------- | -------------- | ----------------------------- |
| `weights`  | <tt>Array</tt> | The model's current weights.  |
| `gradient` | <tt>Array</tt> | The model's current gradient. |
| `lr_scale` | <tt>float</tt> | TODO (default 1.0)            |
| `key`      | <tt>int</tt>   | The model's ID.               |

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

### Optimizer.step_schedules {#step_schedules tag="method"}

Replace the the named hyperparameters with the next item from the schedules
iterator, if available.

```python
### Example
from thinc.api import Optimizer, decaying

schedules = {"learn_rate": decaying(0.001, 1e-4)}
optimizer = Optimizer(learn_rate=0.001, schedules=schedules)
optimizer.step_schedules()
assert optimizer.learn_rate == 0.001
optimizer.step_schedules()
assert optimizer.learn_date == 0.000999900009999
```

### Optimizer.learn_rate {#learn_rate tag="property"}

Get or set the learning rate.

```python
### Example
from thinc.api import Optimizer

optimizer = Optimizer(learn_rate=0.001, L2=1e-6, grad_clip=1.0)
assert optimizer.learn_rate == 0.001
optimizer.learn_rate = 0.01
```
