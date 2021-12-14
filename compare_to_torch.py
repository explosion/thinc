import torch
import numpy as np

from thinc.backends.ops import Ops


def cast_torch(scalar):
    return torch.tensor([scalar],
                        requires_grad=True)


def torch_relu_n(X):
    return torch.nn.functional.relu6(X)


def torch_hard_sigmoid(X):
    return torch.clip(X * 0.2 + 0.5, 0, 1)


def torch_swish(X):
    return torch.nn.functional.silu(X)


def torch_hard_swish(X):
    return X * torch_hard_sigmoid(X)


def torch_hard_swish_mobilenet(X):
    return torch.nn.functional.hardswish(X)


ops = Ops()
PI = cast_torch(np.pi)
ACTIVATION_FUNCTIONS = [(ops.relu_n, ops.backprop_relu_n, torch_relu_n),
                        (ops.hard_sigmoid, ops.backprop_hard_sigmoid, torch_hard_sigmoid),
                        (ops.swish, ops.backprop_swish, torch_swish),
                        (ops.hard_swish_mobilenet, ops.backprop_hard_swish_mobilenet,
                         torch_hard_swish_mobilenet)]


test_scalars = np.random.uniform(-5, 5, (3000))

for forward, backward, pytorch in ACTIVATION_FUNCTIONS:
    print(forward.__name__,
          backward.__name__,
          pytorch.__name__)
    for i in test_scalars:
        x_thinc = np.asarray([i])
        x = cast_torch(i)
        y = pytorch(x)
        y_thinc = forward(x_thinc)
        y.backward()
        assert np.isclose(y_thinc,
                          forward(x_thinc,
                                  inplace=True))
        assert np.isclose(y.detach().numpy(), y_thinc)
        x_thinc = np.asarray([i])
        if backward.__name__ == "backprop_swish":
            assert np.isclose(backward(dY=1, Y=y_thinc, X=x_thinc),
                              backward(dY=1, Y=y_thinc, X=x_thinc, inplace=True))
            assert np.isclose(x.grad.item(), float(backward(dY=1, Y=y_thinc, X=x_thinc)))
        else:
            assert np.isclose(backward(dY=1, X=x_thinc),
                              backward(dY=1, X=x_thinc, inplace=True))
            assert np.isclose(x.grad.item(), float(backward(dY=1, X=x_thinc)))
