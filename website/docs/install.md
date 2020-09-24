---
title: Installation & Setup
---

Thinc is compatible with **64-bit CPython 3.6+** and runs on **Unix/Linux**,
**macOS/OS X** and **Windows**. The latest releases are available from
[pip](https://pypi.python.org/pypi/thinc) and
[conda](https://anaconda.org/conda-forge/thinc). Both installations should come
with binary wheels for Thinc and its dependencies, so you shouldn't have to
compile anything locally.

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

### Run Thinc with GPU {#gpu}

We've been grateful to use the work of Chainer's
[`cupy`](https://cupy.chainer.org) module, which provides a `numpy`-compatible
interface for GPU arrays. Thinc can be installed on GPU by specifying `cuda` and
the optional version identifier in brackets, e.g. `thinc[cuda]` or
`thinc[cuda92]` for CUDA 9.2. If you know your CUDA version, using the more
explicit specifier allows `cupy` to be installed from a wheel, saving some
compilation time.

```bash
### Example
$ pip install -U thinc[cuda92]
```

| CUDA | Install command              | cupy package   |                Wheel                |
| ---- | ---------------------------- | -------------- | :---------------------------------: |
| 8.0  | `pip install thinc[cuda80]`  | `cupy-cuda80`  | <i aria-label="yes" name="yes"></i> |
| 9.0  | `pip install thinc[cuda90]`  | `cupy-cuda90`  | <i aria-label="yes" name="yes"></i> |
| 9.1  | `pip install thinc[cuda91]`  | `cupy-cuda91`  | <i aria-label="yes" name="yes"></i> |
| 9.2  | `pip install thinc[cuda92]`  | `cupy-cuda92`  | <i aria-label="yes" name="yes"></i> |
| 10.0 | `pip install thinc[cuda100]` | `cupy-cuda100` | <i aria-label="yes" name="yes"></i> |
| 10.1 | `pip install thinc[cuda101]` | `cupy-cuda101` | <i aria-label="yes" name="yes"></i> |
| n/a  | `pip install thinc[cuda]`    | `cupy-cuda`    |  <i aria-label="no" name="no"></i>  |

Once you have a GPU-enabled installation, the best way to activate it is to call
[`prefer_gpu`](/docs/api-util#prefer_gpu) (will use GPU if available) or
[`require_gpu`](/docs/api-util#require_gpu) (will raise an error if no GPU is
available).

```python
from thinc.api import prefer_gpu
is_gpu = prefer_gpu()
```

---

## Working with type annotations {#type-annotations}

Thinc makes extensive use of
[type hints](https://docs.python.org/3/library/typing.html) and includes various
[custom types](/docs/api-types) for input and output types, like arrays of
different shapes. This lets you type check your code and model definitions, and
will show you errors if your inputs and outputs don't match, greatly reducing
time spent debugging. To use type checking, you can install
[`mypy`](https://mypy.readthedocs.io/en/stable/) alongside Thinc. If you're
using an editor like Visual Studio Code, you can also
[enable `mypy` linting](https://code.visualstudio.com/docs/python/linting) to
get real-time feedback as you write code.

<!-- TODO: more details and examples (code example of `mypy` output, screenshot of
VSCode linting) -->
