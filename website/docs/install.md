---
title: Installation & Setup
next: /docs/usage-config
---

Thinc is compatible with **64-bit CPython 3.6+** and runs on **Unix/Linux**,
**macOS/OS X** and **Windows**. The latest releases are available from
[pip](https://pypi.python.org/pypi/thinc) and
[conda](https://anaconda.org/conda-forge/thinc).

<grid>

```bash
### pip
$ pip install thinc
```

```bash
### conda
$ conda install -c conda-forge thinc
```

</grid>

<quickstart title="Extended installation" id="extended" suffix=""></quickstart>

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

### Using build constraints when compiling from source

If you install Thinc from source or with `pip` for platforms where there are not
binary wheels on PyPI (currently any non-`x86_64` platforms, so commonly Linux
`aarch64` or OS X M1/`arm64`), you may need to use build constraints if any
package in your environment requires an older version of `numpy`.

If `numpy` gets downgraded from the most recent release at any point after
you've compiled `thinc`, you might see an error that looks like this:

```none
numpy.ndarray size changed, may indicate binary incompatibility.
```

To fix this, create a new virtual environment and install `thinc` and all of its
dependencies using build constraints.
[Build constraints](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
specify an older version of `numpy` that is only used while compiling `thinc`,
and then your runtime environment can use any newer version of `numpy` and still
be compatible. In addition, use `--no-cache-dir` to ignore any previously cached
wheels so that all relevant packages are recompiled from scratch:

```shell
PIP_CONSTRAINT=https://raw.githubusercontent.com/explosion/thinc/master/build-constraints.txt \
pip install thinc --no-cache-dir
```

Our build constraints currently specify the oldest supported `numpy` available
on PyPI for `x86_64`. Depending on your platform and environment, you may want
to customize the specific versions of `numpy`. For other platforms, you can have
a look at SciPy's
[`oldest-supported-numpy`](https://github.com/scipy/oldest-supported-numpy/blob/main/setup.cfg)
package to see what the oldest recommended versions of `numpy` are.

(_Warning_: don't use `pip install -c constraints.txt` instead of
`PIP_CONSTRAINT`, since this isn't applied to the isolated build environments.)

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
