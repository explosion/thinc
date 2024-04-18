import numpy
import pytest

from thinc.compat import has_apple_ops

try:
    import thinc.backends._accelerate as accelerate
except:
    pass


@pytest.mark.skipif(not has_apple_ops, reason="Apple ops not available")
def test_basic_sgemm():
    A = numpy.random.randn(5, 4).astype("f")
    B = numpy.random.randn(4, 7).astype("f")
    C = accelerate.gemm(A, B)
    assert C.shape == (A.shape[0], B.shape[1])

    C_out = numpy.empty((5, 7), dtype="f")
    accelerate.gemm(A, B, out=C_out)

    numpy.testing.assert_allclose(C, C_out)


@pytest.mark.skipif(not has_apple_ops, reason="Apple ops not available")
def test_incorrect_output_size():
    A = numpy.ndarray((5, 4), dtype="f")
    B = numpy.ndarray((4, 7), dtype="f")

    with pytest.raises(ValueError, match=r"Shape mismatch for output matrix"):
        accelerate.gemm(A, B, out=numpy.ndarray((3, 7), dtype="f"))

    with pytest.raises(ValueError, match=r"Shape mismatch for output matrix"):
        accelerate.gemm(A, B, out=numpy.ndarray((5, 3), dtype="f"))


@pytest.mark.skipif(not has_apple_ops, reason="Apple ops not available")
@pytest.mark.parametrize(
    "A_shape,B_shape,transA,transB",
    [
        [(0, 0), (0, 0), False, False],
        [(0, 0), (0, 0), True, False],
        [(0, 0), (0, 0), False, True],
        [(0, 0), (0, 0), True, True],
        [(0, 5), (5, 0), False, False],
        [(5, 0), (5, 0), False, True],
        [(5, 0), (5, 0), True, False],
    ],
)
def test_zero_size(A_shape, B_shape, transA, transB):
    A = numpy.ndarray(A_shape, dtype="f")
    B = numpy.ndarray(B_shape, dtype="f")
    if not transA and not transB:
        C = numpy.dot(A, B)
    elif transA:
        C = numpy.dot(A.T, B)
    elif transB:
        C = numpy.dot(A, B.T)
    else:
        C = numpy.dot(A.T, B.T)
    C_ = accelerate.gemm(A, B, trans1=transA, trans2=transB)
    assert C.shape == C_.shape


@pytest.mark.skipif(not has_apple_ops, reason="Apple ops not available")
@pytest.mark.parametrize(
    "A_shape,B_shape,transA,transB",
    [
        [(4, 5), (4, 5), False, False],
        [(5, 4), (4, 5), True, False],
        [(4, 5), (5, 4), False, True],
        [(5, 4), (5, 4), True, True],
    ],
)
def test_incorrect_shapes(A_shape, B_shape, transA, transB):
    A = numpy.ndarray(A_shape, dtype="f")
    B = numpy.ndarray(B_shape, dtype="f")
    with pytest.raises(ValueError, match=r"Shape mismatch"):
        accelerate.gemm(A, B, trans1=transA, trans2=transB)
