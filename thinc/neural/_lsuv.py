import numpy as np

# Layer-sequential Unit Variance initialization, by
# https://github.com/ducha-aiki/LSUV-keras/blob/master/lsuv_init.py


# Orthonorm init code is taken from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(shape):
    if len(shape) < 2: # pragma: no cover
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def do_lsuv(ops, weights, predict, X):
    weights[:] = svd_orthonormal(weights.shape)
    acts = predict(X)
    tol_var = 0.1
    t_max = 10
    t_i = 0
    while True:
        acts1 = predict(X)
        var = np.var(acts1)
        if abs(var - 1.0) < tol_var or t_i > t_max:
            break
        weights /= ops.xp.sqrt(var)
        t_i += 1
    return predict(X)


def LSUVinit(model, X, y=None):
    if model.name == 'batchnorm': # pragma: no cover
        model = model._layers[0]
    if model.name in 'softmax': # pragma: no cover
        return
    model.W[:] = svd_orthonormal(model.W.shape)
    acts = model(X)
    tol_var = 0.1
    t_max = 10
    t_i = 0
    while True:
        acts1 = model(X)
        var = np.var(acts1)
        if abs(var - 1.0) < tol_var or t_i > t_max:
            break
        model.W /= model.ops.xp.sqrt(var)
        t_i += 1
    acts = model(X)
