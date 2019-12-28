from .neural.util import get_array_module, copy_array


def xavier_uniform_init(W, inplace=False):
    xp = get_array_module(W)
    scale = xp.sqrt(6. / (W.shape[0] + W.shape[1]))
    if inplace:
        copy_array(W, xp.random.uniform(-scale, scale, W.shape))
        return W
    else:
        return xp.random.uniform(-scale, scale, W.shape)


def zero_init(data, inplace=False):
    if inplace:
        data.fill(0.)
        return data
    else:
        xp = get_array_module(data)
        return xp.zeroslike(data)


def uniform_init(data, lo=-0.1, hi=0.1, inplace=False):
    xp = get_array_module(data)
    values = xp.random.uniform(lo, hi, data.shape)
    if inplace:
        copy_array(data, values)
    else:
        return values.astype(data.dtype)
