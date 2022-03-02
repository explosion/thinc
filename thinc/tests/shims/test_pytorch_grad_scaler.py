import pytest

from hypothesis import given, settings
from hypothesis.strategies import lists, one_of, tuples
from thinc.util import has_torch, has_torch_gpu, is_torch_array
from thinc.api import PyTorchGradScaler

from ..strategies import ndarrays


def tensors():
    # This function is not used without Torch + CUDA,
    # but we have to do some wrapping to avoid import
    # failures.
    try:
        import torch

        return ndarrays().map(lambda a: torch.tensor(a).cuda())
    except ImportError:
        pass


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.skipif(not has_torch_gpu, reason="needs a GPU")
@given(X=one_of(tensors(), lists(tensors()), tuples(tensors())))
@settings(deadline=None)
def test_scale_random_inputs(X):
    import torch

    device_id = torch.cuda.current_device()
    scaler = PyTorchGradScaler(enabled=True)
    scaler.to_(device_id)

    if is_torch_array(X):
        assert torch.allclose(scaler.scale(X), X * 2.0 ** 16)
    else:
        scaled1 = scaler.scale(X)
        scaled2 = [t * 2.0 ** 16 for t in X]
        for t1, t2 in zip(scaled1, scaled2):
            assert torch.allclose(t1, t2)


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.skipif(not has_torch_gpu, reason="needs a GPU")
def test_grad_scaler():
    import torch

    device_id = torch.cuda.current_device()

    scaler = PyTorchGradScaler(enabled=True)
    scaler.to_(device_id)

    #  Test that scaling works as expected.
    t = torch.tensor([1.0], device=device_id)
    assert scaler.scale([torch.tensor([1.0], device=device_id)]) == [
        torch.tensor([2.0 ** 16], device=device_id)
    ]
    assert scaler.scale(torch.tensor([1.0], device=device_id)) == torch.tensor(
        [2.0 ** 16], device=device_id
    )
    with pytest.raises(ValueError):
        scaler.scale("bogus")
    with pytest.raises(ValueError):
        scaler.scale(42)

    # Test infinity detection.
    g = [
        torch.tensor([2.0 ** 16], device=device_id),
        torch.tensor([float("Inf")], device=device_id),
    ]

    # Check that infinity was found.
    assert scaler.unscale(g)

    # Check whether unscale was successful.
    assert g[0] == torch.tensor([1.0]).cuda()

    scaler.update()

    # Since infinity was found, the scale should be halved from 2**16
    # to 2**15 for the next step.
    assert scaler.scale([torch.tensor([1.0], device=device_id)]) == [
        torch.tensor([2.0 ** 15], device=device_id)
    ]
