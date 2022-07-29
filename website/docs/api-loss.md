---
title: Loss Calculators
next: /docs/api-config
---

All loss calculators follow the same API: they're classes that are initialized
with optional settings and have a `get_grad` method returning the gradient of
the loss with respect to the model outputs and a `get_loss` method returning the
scalar loss.

## Loss {#loss tag="base class"}

### Loss.\_\_init\_\_ {#loss-init tag="method"}

Initialize the loss calculator.

| Argument   | Type         | Description                                                                              |
| ---------- | ------------ | ---------------------------------------------------------------------------------------- |
| `**kwargs` | <tt>Any</tt> | Optional calculator-specific settings. Can also be provided via the [config](#registry). |

### Loss.\_\_call\_\_ {#loss-call tag="method"}

Calculate the gradient and the scalar loss. Returns a tuple of the results of
`Loss.get_grad` and `Loss.get_loss`.

| Argument    | Type                     | Description                   |
| ----------- | ------------------------ | ----------------------------- |
| `guesses`   | <tt>Any</tt>             | The model outputs.            |
| `truths`    | <tt>Any</tt>             | The training labels.          |
| **RETURNS** | <tt>Tuple[Any, Any]</tt> | The gradient and scalar loss. |

### Loss.get_grad {#loss-get_grad tag="method"}

Calculate the gradient of the loss with respect with the model outputs.

| Argument    | Type         | Description          |
| ----------- | ------------ | -------------------- |
| `guesses`   | <tt>Any</tt> | The model outputs.   |
| `truths`    | <tt>Any</tt> | The training labels. |
| **RETURNS** | <tt>Any</tt> | The gradient.        |

### Loss.get_loss {#loss-get_grad tag="method"}

Calculate the scalar loss. Typically returns a float.

| Argument    | Type         | Description          |
| ----------- | ------------ | -------------------- |
| `guesses`   | <tt>Any</tt> | The model outputs.   |
| `truths`    | <tt>Any</tt> | The training labels. |
| **RETURNS** | <tt>Any</tt> | The scalar loss.     |

---

## Loss Calculators {#calculators}

### CategoricalCrossentropy {#categorical_crossentropy tag="class"}

<inline-list>

- **Guesses:** <tt>Floats2d</tt>
- **Truths:** <tt>Union[Ints1d, List[int], List[str], Floats2d]</tt>
- **Gradient:** <tt>Floats2d</tt>
- **Loss:** <tt>float</tt>

</inline-list>

A flexible implementation of the common categorical cross-entropy loss that
works on various data types. The `guesses` should represent probabilities and
are expected to be in the range of `[0, 1]`. They can both represent exclusive
classes from multi-class cross-entropy (generally coming from a `softmax` layer)
or could be classwise binary decisions for multi-label cross-entropy (`sigmoid`
layer). The `truths` are most commonly provided as labels in `Ints1d`,
`List[int]` or `List[str]` format. Alternatively, users can provide `truths` as
a `Floats2d` for example to encode label-confidences.

<grid>

```python
### {small="true"}
from thinc.api import CategoricalCrossentropy
loss_calc = CategoricalCrossentropy()
```

```ini
### config.cfg {small="true"}
[loss]
@losses = "CategoricalCrossentropy.v1"
normalize = true
```

</grid>

| Argument          | Type                     |  Description                                                                                                                                     |
| ----------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| _keyword-only_    |                          |                                                                                                                                                  |
| `names`           | <tt>List[str]</tt>       | Label names. Has to be provided when using with List[str] as truths.                                                                             |
| `normalize`       | <tt>bool</tt>            | Normalize and divide by number of examples given.                                                                                                |
| `neg_prefix`      | <tt>str</tt>             | Prefix used to indicate that a label is negative e.g. "!sci-fi".                                                                                 |
| `missing_value`   | <tt>Union[str, int]</tt> | Specific label that indicates the value is missing and should not be considered for training/evaluation purposes, e.g. empty string `""` or `0`. |
| `label_smoothing` | <tt>float</tt>           | Smoothing-coefficient for label-smoothing.                                                                                                       |

### SequenceCategoricalCrossentropy {#sequence_categorical_crossentropy tag="class"}

<inline-list>

- **Guesses:** <tt>List[Floats2d]</tt>
- **Truths:** <tt>List[Union[Ints1d, List[int], List[str], Floats2d]]</tt>
- **Gradient:** <tt>List[Floats2d]</tt>
- **Loss:** <tt>List[float]</tt>

</inline-list>

This loss runs the `CategoricalCrossentropy` over a `List` of `guesses` and
`truths`.

<grid>

```python
### {small="true"}
from thinc.api import SequenceCategoricalCrossentropy
loss_calc = SequenceCategoricalCrossentropy()
```

```ini
### config.cfg {small="true"}
[loss]
@losses = "SequenceCategoricalCrossentropy.v1"
normalize = true
```

</grid>

| Argument          | Type                     |  Description                                                        |
| ----------------- | ------------------------ | ------------------------------------------------------------------- |
| _keyword-only_    |                          |                                                                     |
| `names`           | <tt>List[str]</tt>       | Label names. Has to be provided when using with List[str] as truths |
| `normalize`       | <tt>bool</tt>            | Normalize and divide by number of examples given.                   |
| `neg_prefix`      | <tt>str</tt>             | Symbol that indicates that a label is negative e.g. "!sci-fi".      |
| `missing_value`   | <tt>Union[str, int]</tt> | Symbol for "missing value" among the labels.                        |
| `label_smoothing` | <tt>float</tt>           | Smoothing-coefficient for label-smoothing.                          |

### L2Distance {#l2distance tag="class"}

<inline-list>

- **Guesses:** <tt>Floats2d</tt>
- **Truths:** <tt>Floats2d</tt>
- **Gradient:** <tt>Floats2d</tt>
- **Loss:** <tt>float</tt>

</inline-list>

<grid>

```python
### {small="true"}
from thinc.api import L2Distance
loss_calc = L2Distance()
```

```ini
### config.cfg {small="true"}
[loss]
@losses = "L2Distance.v1"
normalize = true
```

</grid>

| Argument       | Type          |  Description                                      |
| -------------- | ------------- | ------------------------------------------------- |
| _keyword-only_ |               |                                                   |
| `normalize`    | <tt>bool</tt> | Normalize and divide by number of examples given. |

### CosineDistance {#cosine_distance tag="function"}

<inline-list>

- **Guesses:** <tt>Floats2d</tt>
- **Truths:** <tt>Floats2d</tt>
- **Gradient:** <tt>Floats2d</tt>
- **Loss:** <tt>float</tt>

</inline-list>

<grid>

```python
### {small="true"}
from thinc.api import CosineDistance
loss_calc = CosineDistance(ignore_zeros=False)
```

```ini
### config.cfg {small="true"}
[loss]
@losses = "CosineDistance.v1"
normalize = true
ignore_zeros = false
```

</grid>

| Argument       | Type          |  Description                                      |
| -------------- | ------------- | ------------------------------------------------- |
| _keyword-only_ |               |                                                   |
| `normalize`    | <tt>bool</tt> | Normalize and divide by number of examples given. |
| `ignore_zeros` | <tt>bool</tt> | Don't count zero vectors.                         |

---

## Usage via config and function registry {#registry}

Defining the loss calculators in the [config](/docs/usage-config) will return
the **initialized object**. Within your script, you can then call it or its
methods and pass in the data.

<grid>

```ini
### config.cfg {small="true"}
[loss]
@losses = "L2Distance.v1"
normalize = true
```

```python
### Usage {small="true"}
from thinc.api import registry, Config

config = Config().from_disk("./config.cfg")
resolved = registry.resolve(config)
loss_calc = resolved["loss"]
loss = loss_calc.get_grad(guesses, truths)
```

</grid>
