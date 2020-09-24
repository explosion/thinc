---
title: Initializers
next: /docs/api-schedules
---

A collection of initialization functions. Parameter initialization schemes can
be very important for deep neural networks, because the initial distribution of
the weights helps determine whether activations change in mean and variance as
the signal moves through the network. If the activations are not stable, the
network will not learn effectively. The "best" initialization scheme changes
depending on the activation functions being used, which is why a variety of
initializations are necessary. You can reduce the importance of the
initialization by using normalization after your hidden layers.

### normal_init {#normal_init tag="function"}

Initialize from a normal distribution, with `scale = sqrt(1 / fan_in)`.

| Argument       | Type           | Description                                |
| -------------- | -------------- | ------------------------------------------ |
| `data`         | <tt>Array</tt> | The array to initialize.                   |
| `fan_in`       | <tt>int</tt>   | Usually the number of inputs to the layer. |
| _keyword-only_ |                |                                            |
| `inplace`      | <tt>bool</tt>  | If `True`, `data` is modified in place.    |
| **RETURNS**    | <tt>Array</tt> | The initialized array.                     |

### xavier_uniform_init {#xavier_uniform_init tag="function"}

Initialize with the randomization introduced by Xavier Glorot
([Glorot and Bengio, 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf),
which is a uniform distribution centered on zero, with
`scale = sqrt(6.0 / (data.shape[0] + data.shape[1]))`. Usually used in ReLu
layers.

| Argument       | Type           | Description                             |
| -------------- | -------------- | --------------------------------------- |
| `data`         | <tt>Array</tt> | The array to initialize.                |
| _keyword-only_ |                |                                         |
| `inplace`      | <tt>bool</tt>  | If `True`, `data` is modified in place. |
| **RETURNS**    | <tt>Array</tt> | The initialized array.                  |

### zero_init {#zero_init tag="function"}

Initialize a parameter with zero weights. This is usually used for output layers
and for bias vectors.

| Argument       | Type           | Description                             |
| -------------- | -------------- | --------------------------------------- |
| `data`         | <tt>Array</tt> | The array to initialize.                |
| _keyword-only_ |                |                                         |
| `inplace`      | <tt>bool</tt>  | If `True`, `data` is modified in place. |
| **RETURNS**    | <tt>Array</tt> | The initialized array.                  |

### uniform_init {#uniform_init tag="function"}

Initialize values from a uniform distribution. This is usually used for word
embedding tables.

| Argument       | Type           | Description                              |
| -------------- | -------------- | ---------------------------------------- |
| `data`         | <tt>Array</tt> | The array to initialize.                 |
| `lo`           | <tt>float</tt> | The minimum of the uniform distribution. |
| `hi`           | <tt>float</tt> | The maximum of the uniform distribution. |
| _keyword-only_ |                |                                          |
| `inplace`      | <tt>bool</tt>  | If `True`, `data` is modified in place.  |
| **RETURNS**    | <tt>Array</tt> | The initialized array.                   |
