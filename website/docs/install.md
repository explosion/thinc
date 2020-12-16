---
title: Installation & Setup
next: /docs/usage-config
---

Thinc is compatible with **64-bit CPython 3.6+** and runs on **Unix/Linux**,
**macOS/OS X** and **Windows**. The latest releases with binary wheels are
available from [pip](https://pypi.python.org/pypi/thinc). For the most recent
releases, pip 19.3 or newer is recommended.

```bash
### pip
$ pip install thinc --pre
```

<infobox variant="warning">

Note that Thinc 8.0 is currently **in alpha preview** and not necessarily ready
for production yet.

</infobox>

<!--The latest releases are available from
[pip](https://pypi.python.org/pypi/thinc) and
[conda](https://anaconda.org/conda-forge/thinc). Both installations should come
with binary wheels for Thinc and its dependencies, so you shouldn't have to
compile anything locally.

<grid>

```bash
### pip
$ pip install thinc
```

<!-- ```bash
### conda
$ conda install -c conda-forge thinc
``` -->

</grid>

<quickstart title="Extended installation" id="extended" suffix=" --pre"></quickstart>

<infobox variant="warning">

If you have installed PyTorch and you are using Python 3.7+, uninstall the
package `dataclasses` with `pip uninstall dataclasses`, since it may have been
installed by PyTorch and is incompatible with Python 3.7+.

</infobox>

If you know your CUDA version, using the more explicit specifier allows `cupy`
to be installed from a wheel, saving some compilation time. Once you have a
GPU-enabled installation, the best way to activate it is to call
[`prefer_gpu`](/docs/api-util#prefer_gpu) (will use GPU if available) or
[`require_gpu`](/docs/api-util#require_gpu) (will raise an error if no GPU is
available).

```python
from thinc.api import prefer_gpu
is_gpu = prefer_gpu()
```

---

## Set up static type checking {#type-checking}

Thinc makes extensive use of
[type hints](https://docs.python.org/3/library/typing.html) and includes various
[custom types](/docs/api-types) for input and output types, like arrays of
different shapes. This lets you type check your code and model definitions, and
will show you errors if your inputs and outputs don't match, greatly reducing
time spent debugging. To use type checking, you can install
[`mypy`](https://mypy.readthedocs.io/en/stable/) alongside Thinc. If you're
using an editor like Visual Studio Code, you can also
[enable `mypy` linting](https://code.visualstudio.com/docs/python/linting) to
get real-time feedback as you write code. For more details, check out the docs
on [using type checking](/docs/usage-type-checking).

<grid>

```bash
### pip {small="true"}
$ pip install mypy
```

```ini
### mypy.ini {small="true"}
[mypy]
plugins = thinc.mypy
```

</grid>

<code-screenshot>

![Screenshot of mypy linting in Visual Studio Code](images/type_checking2.jpg)

</code-screenshot>
