import pytest

from thinc.util import has_torch, has_torch_gpu

from thinc.api import PyTorchGradScaler


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

    # Test infinity detection.
    g = [
        torch.tensor([2.0 ** 16], device=device_id),
        torch.tensor([float("Inf")], device=device_id),
    ]

    # Check that infinity was found.
    assert scaler.unscale(g)

    # Check whether unscale was succesful.
    assert g[0] == torch.tensor([1.0]).cuda()

    scaler.update()

    # Since infinity was found, the scale should be halved from 2**16
    # to 2**15 for the next step.
    assert scaler.scale([torch.tensor([1.0], device=device_id)]) == [
        torch.tensor([2.0 ** 15], device=device_id)
    ]
