from ..util import tensorflow2xp, assert_tensorflow_installed

try:
    import tensorflow
except ImportError:
    pass


def cupy_tensorflow_allocator(size_in_bytes: int):
    assert_tensorflow_installed()
    tensor = tensorflow.zeros((size_in_bytes // 4,), dtype=tensorflow.dtypes.float32)
    # We convert to cupy via dlpack, so that we can get a memory pointer.
    cupy_array = tensorflow2xp(tensor)
    # Now return the array's memory pointer.
    return cupy_array.ptr  # type: ignore
