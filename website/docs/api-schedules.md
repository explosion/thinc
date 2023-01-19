---
title: Schedules
next: /docs/api-loss
---

Schedules are generators that provide different rates, schedules, decays or
series. They're typically used for batch sizes or learning rates. You can easily
implement your own schedules as well: just write your own
[`Schedule`](#schedule) implementation, that produces whatever series of values
you need. A common use case for schedules is within
[`Optimizer`](/docs/api-optimizer) objects, which accept iterators for most of
their parameters. See the [training guide](/docs/usage-training) for details.

## Schedule {#schedule tag="class" new="9"}

Class for implementing Thinc schedules.

<infobox variant="warning">

There's only one `Schedule` class in Thinc and schedules are built using
**composition**, not inheritance. This means that a schedule or composed
schedule will return an **instance** of `Schedule` – it doesn't subclass it. To
read more about this concept, see the pages on
[Thinc's philosophy](/docs/concept).

</infobox>

### Typing {#typing}

`Schedule` can be used as a
[generic type](https://docs.python.org/3/library/typing.html#generics) with one
parameter. This parameter specifies the type that is returned by the schedule.
For instance, `Schedule[int]` denotes a scheduler that returns integers when
called. A mismatch will cause a type error. For more details, see the docs on
[type checking](/docs/usage-type-checking).

```python
from thinc.api import Schedule

def my_function(schedule: Schedule[int]):
    ...
```

### Attributes {#attributes}

| Name   | Type         | Description                     |
| ------ | ------------ | ------------------------------- |
| `name` | <tt>str</tt> | The name of the scheduler type. |

### Properties {#properties}

| Name    | Type                    | Description                                                                                                                                                               |
| ------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attrs` | <tt>Dict[str, Any]</tt> | The scheduler attributes. You can use the dict directly and assign _to_ it – but you cannot reassign `schedule.attrs` to a new variable: `schedule.attrs = {}` will fail. |

### Schedule.\_\_init\_\_ {#init tag="method"}

Initialize a new schedule.

```python
### Example
schedule = Schedule(
    "constant",
    constant_schedule,
    attrs={"rate": rate},
)
```

| Argument       | Type                    | Description                                              |
| -------------- | ----------------------- | -------------------------------------------------------- |
| `name`         | <tt>str</tt>            | The name of the schedule type.                           |
| `schedule`     | <tt>Callable</tt>       | Function to compute the schedule value for a given step. |
| _keyword-only_ |                         |                                                          |
| `attrs`        | <tt>Dict[str, Any]</tt> | Dictionary of non-parameter attributes.                  |

### Schedule.\_\_call\_\_ {#call tag="method"}

Call the schedule function, returning the value for the given step. The `step`
positional argument is always required. Some schedules may require additional
keyword arguments.

```python
### Example
from thinc.api import constant

schedule = constant(0.1)
assert schedule(0) == 0.1
assert schedule(1000) == 0.1
```

| Argument    | Type         | Description                                |
| ----------- | ------------ | ------------------------------------------ |
| `step`      | <tt>int</tt> | The step to compute the schedule for.      |
| `**kwargs`  |              | Optional arguments passed to the schedule. |
| **RETURNS** | <tt>Any</tt> | The schedule value for the step.           |

### Schedule.to_generator {#to_generator tag="method"}

Turn the schedule into a generator by passing monotonically increasing step
count into the schedule.

```python
### Example
from thinc.api import constant

g = constant(0.1).to_generator()
assert next(g) == 0.1
assert next(g) == 0.1
```

| Argument    | Type                                 | Description                                                                     |
| ----------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| `start`     | <tt>int</tt>                         | The initial schedule step. Defaults to `0`.                                     |
| `step_size` | <tt>int</tt>                         | The amount to increase the step with for each generated value. Defaults to `1`. |
| `**kwargs`  |                                      | Optional arguments passed to the schedule.                                      |
| **RETURNS** | <tt>Generator[OutT, None, None]</tt> | The generator.                                                                  |

## constant {#constant tag="function"}

Yield a constant rate.

![](images/schedules_constant.svg)

<grid>

```python
### {small="true"}
from thinc.api import constant

batch_sizes = constant(0.001)
batch_size = batch_sizes(step=0)
```

```ini
### config {small="true"}
[batch_size]
@schedules = "constant.v1"
rate = 0.001
```

</grid>

| Argument   | Type           |
| ---------- | -------------- |
| `rate`     | <tt>float</tt> |
| **YIELDS** | <tt>float</tt> |

## constant_then {#constant_then tag="function"}

Yield a constant rate for N steps, before starting a schedule.

![](images/schedules_constant_then.svg)

<grid>

```python
### {small="true"}
from thinc.api import constant_then, decaying

learn_rates = constant_then(
    0.005,
    1000,
    decaying(0.005, 1e-4)
)
learn_rate = learn_rates(step=0)
```

```ini
### config {small="true"}
[learn_rates]
@schedules = "constant_then.v1"
rate = 0.005
steps = 1000

[learn_rates.schedule]
@schedules = "decaying"
base_rate = 0.005
decay = 1e-4
```

</grid>

| Argument   | Type                     |
| ---------- | ------------------------ |
| `rate`     | <tt>float</tt>           |
| `steps`    | <tt>int</tt>             |
| `schedule` | <tt>Iterable[float]</tt> |
| **YIELDS** | <tt>float</tt>           |

## decaying {#decaying tag="function"}

Yield an infinite series of linearly decaying values, following the schedule
`base_rate * 1 / (1 + decay * t)`.

![](images/schedules_decaying.svg)

<grid>

```python
### {small="true"}
from thinc.api import decaying

learn_rates = decaying(0.005, 1e-4)
learn_rate = learn_rates(step=0)  # 0.001
learn_rate = learn_rates(step=1)  # 0.00999
```

```ini
### config {small="true"}
[learn_rate]
@schedules = "decaying.v1"
base_rate = 0.005
decay = 1e-4
t = 0
```

</grid>

| Argument       | Type           |
| -------------- | -------------- |
| `base_rate`    | <tt>float</tt> |
| `decay`        | <tt>float</tt> |
| _keyword-only_ |                |
| `t`            | <tt>int</tt>   |
| **YIELDS**     | <tt>float</tt> |

## compounding {#compounding tag="function"}

Yield an infinite series of compounding values. Each time the generator is
called, a value is produced by multiplying the previous value by the compound
rate.

![](images/schedules_compounding.svg)

<grid>

```python
### {small="true"}
from thinc.api import compounding

batch_sizes = compounding(1.0, 32.0, 1.001)
batch_size = batch_sizes(step=0)  # 1.0
batch_size = batch_sizes(step=1)  # 1.0 * 1.001
```

```ini
### config {small="true"}
[batch_size]
@schedules = "compounding.v1"
start = 1.0
stop = 32.0
compound = 1.001
t = 0
```

</grid>

| Argument       | Type           |
| -------------- | -------------- |
| `start`        | <tt>float</tt> |
| `stop`         | <tt>float</tt> |
| `compound`     | <tt>float</tt> |
| _keyword-only_ |                |
| `t`            | <tt>int</tt>   |
| **YIELDS**     | <tt>float</tt> |

## warmup_linear {#warmup_linear tag="function"}

Generate a series, starting from an initial rate, and then with a warmup period,
and then a linear decline. Used for learning rates.

![](images/schedules_warmup_linear.svg)

<grid>

```python
### {small="true"}
from thinc.api import warmup_linear

learn_rates = warmup_linear(0.01, 3000, 6000)
learn_rate = learn_rates(step=0)
```

```ini
### config {small="true"}
[learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.01
warmup_steps = 3000
total_steps = 6000
```

</grid>

| Argument       | Type           |
| -------------- | -------------- |
| `initial_rate` | <tt>float</tt> |
| `warmup_steps` | <tt>int</tt>   |
| `total_steps`  | <tt>int</tt>   |
| **YIELDS**     | <tt>float</tt> |

## slanted_triangular {#slanted_triangular tag="function"}

Yield an infinite series of values according to
[Howard and Ruder's (2018)](https://arxiv.org/abs/1801.06146) "slanted
triangular learning rate" schedule.

![](images/schedules_slanted_triangular.svg)

<grid>

```python
### {small="true"}
from thinc.api import slanted_triangular

learn_rates = slanted_triangular(0.1, 5000)
learn_rate = learn_rates(step=0)
```

```ini
### config {small="true"}
[learn_rate]
@schedules = "slanted_triangular.v1"
max_rate = 0.1
num_steps = 5000
cut_frac = 0.1
ratio = 32
decay = 1.0
t = 0.1
```

</grid>

| Argument       | Type           |
| -------------- | -------------- |
| `max_rate`     | <tt>float</tt> |
| `num_steps`    | <tt>int</tt>   |
| _keyword-only_ |                |
| `cut_frac`     | <tt>float</tt> |
| `ratio`        | <tt>int</tt>   |
| `decay`        | <tt>float</tt> |
| `t`            | <tt>float</tt> |
| **YIELDS**     | <tt>float</tt> |

## cyclic_triangular {#cyclic_triangular tag="function"}

Linearly increasing then linearly decreasing the rate at each cycle.

![](images/schedules_cyclic_triangular.svg)

<grid>

```python
### {small="true"}
from thinc.api import cyclic_triangular

learn_rates = cyclic_triangular(0.005, 0.001, 1000)
learn_rate = learn_rates(step=0)
```

```ini
### config {small="true"}
[learn_rate]
@schedules = "cyclic_triangular.v1"
min_lr = 0.005
max_lr = 0.001
period = 1000
```

</grid>

| Argument   | Type           |
| ---------- | -------------- |
| `min_lr`   | <tt>float</tt> |
| `max_lr`   | <tt>float</tt> |
| `period`   | <tt>int</tt>   |
| **YIELDS** | <tt>float</tt> |

## plateau {#plateau tag="function" new="9"}

Yields values from the wrapped schedule, exponentially scaled by the number of
times optimization has plateaued. The caller must pass model evaluation scores
through the `last_score` argument for the scaling to be adjusted. The last
evaluation score is passed through the `last_score` argument as a tuple
(`last_score_step`, `last_score`). This tuple indicates when a model was last
evaluated (`last_score_step`) and with what score (`last_score`).

<grid>

```python
### {small="true"}
from thinc.api import constant, plateau

schedule = plateau(2, 0.5, constant(1.0))
assert schedule(step=0, last_score=(0, 1.0)) == 1.0
assert schedule(step=1, last_score=(1, 1.0)) == 1.0
assert schedule(step=2, last_score=(2, 1.0)) == 0.5
assert schedule(step=3, last_score=(3, 1.0)) == 0.5
assert schedule(step=4, last_score=(4, 1.0)) == 0.25
```

```ini
### config {small="true"}
[learn_rate]
@schedules = "plateau.v1"
scale = 0.5
max_patience = 2

[learn_rate.shedule]
@schedules = "constant.v1"
rate = 1.0
```

</grid>

| Argument       | Type                     | Description                                                                           |
| -------------- | ------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `max_patience` | <tt>int</tt>             | Number of evaluations without an improvement to consider the model to have plateaued. |
| `scale`        | <tt>float</tt>           |                                                                                       | Scaling of the inner schedule after plateauing. |
| `schedule`     | <tt>Schedule[float]</tt> |                                                                                       | The schedule to wrap.                           |
| **RETURNS**    | <tt>Schedule[float]</tt> |                                                                                       |
