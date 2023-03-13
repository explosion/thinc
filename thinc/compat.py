from packaging.version import Version

try:  # pragma: no cover
    import cupy
    import cupyx

    has_cupy = True
    cupy_version = Version(cupy.__version__)
    try:
        cupy.cuda.runtime.getDeviceCount()
        _has_cupy_gpu = True
    except cupy.cuda.runtime.CUDARuntimeError:
        _has_cupy_gpu = False

    if cupy_version.major >= 10:
        # fromDlpack was deprecated in v10.0.0.
        cupy_from_dlpack = cupy.from_dlpack
    else:
        cupy_from_dlpack = cupy.fromDlpack
except (ImportError, AttributeError):
    cupy = None
    cupyx = None
    cupy_version = Version("0.0.0")
    has_cupy = False
    cupy_from_dlpack = None
    _has_cupy_gpu = False


try:  # pragma: no cover
    import torch.utils.dlpack
    import torch

    _has_torch = True
    _has_torch_cuda_gpu = torch.cuda.device_count() != 0
    _has_torch_mps_gpu = (
        hasattr(torch, "has_mps")
        and torch.has_mps  # type: ignore[attr-defined]
        and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    )
    _has_torch_gpu = _has_torch_cuda_gpu
    torch_version = Version(str(torch.__version__))
    _has_torch_amp = (
        torch_version >= Version("1.9.0")
        and not torch.cuda.amp.common.amp_definitely_not_available()
    )
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _has_torch = False
    _has_torch_cuda_gpu = False
    _has_torch_gpu = False
    _has_torch_mps_gpu = False
    _has_torch_amp = False
    torch_version = Version("0.0.0")

try:  # pragma: no cover
    import tensorflow.experimental.dlpack
    import tensorflow

    _has_tensorflow = True
    _has_tensorflow_gpu = len(tensorflow.config.get_visible_devices("GPU")) > 0
except ImportError:  # pragma: no cover
    tensorflow = None
    _has_tensorflow = False
    _has_tensorflow_gpu = False


try:  # pragma: no cover
    import mxnet

    _has_mxnet = True
except ImportError:  # pragma: no cover
    mxnet = None
    _has_mxnet = False

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


try:  # pragma: no cover
    import os_signpost

    _has_os_signpost = True
except ImportError:
    os_signpost = None
    _has_os_signpost = False


try:  # pragma: no cover
    import blis

    _has_blis = True
except ImportError:
    blis = None
    _has_blis = False


_has_gpu = _has_cupy_gpu or _has_torch_mps_gpu

__all__ = [
    "cupy",
    "cupyx",
    "torch",
    "tensorflow",
    "mxnet",
    "h5py",
    "os_signpost",
]
