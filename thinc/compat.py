from packaging.version import Version

try:  # pragma: no cover
    import cupy
    import cupyx

    has_cupy = True
    cupy_version = Version(cupy.__version__)
    try:
        cupy.cuda.runtime.getDeviceCount()
        has_cupy_gpu = True
    except cupy.cuda.runtime.CUDARuntimeError:
        has_cupy_gpu = False

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
    has_cupy_gpu = False


try:  # pragma: no cover
    import torch.utils.dlpack
    import torch

    has_torch = True
    has_torch_cuda_gpu = torch.cuda.device_count() != 0
    has_torch_mps_gpu = (
        hasattr(torch, "has_mps")
        and torch.has_mps  # type: ignore[attr-defined]
        and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    )
    has_torch_gpu = has_torch_cuda_gpu
    torch_version = Version(str(torch.__version__))
    has_torch_amp = (
        torch_version >= Version("1.9.0")
        and not torch.cuda.amp.common.amp_definitely_not_available()
    )
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    has_torch = False
    has_torch_cuda_gpu = False
    has_torch_gpu = False
    has_torch_mps_gpu = False
    has_torch_amp = False
    torch_version = Version("0.0.0")

try:  # pragma: no cover
    import tensorflow.experimental.dlpack
    import tensorflow

    has_tensorflow = True
    has_tensorflow_gpu = len(tensorflow.config.get_visible_devices("GPU")) > 0
except ImportError:  # pragma: no cover
    tensorflow = None
    has_tensorflow = False
    has_tensorflow_gpu = False


try:  # pragma: no cover
    import mxnet

    has_mxnet = True
except ImportError:  # pragma: no cover
    mxnet = None
    has_mxnet = False

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


try:  # pragma: no cover
    import os_signpost

    has_os_signpost = True
except ImportError:
    os_signpost = None
    has_os_signpost = False


has_gpu = has_cupy_gpu or has_torch_mps_gpu
