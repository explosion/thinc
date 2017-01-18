import numpy as np

# Layer-sequential Unit Variance initialization, by
# https://github.com/ducha-aiki/LSUV-keras/blob/master/lsuv_init.py


# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def LSUVinit(model, X, y=None):
    if model.name == 'softmax':
        return
    model.W[:] = svd_orthonormal(model.W.shape)
    acts = model(X)
    print(model.name, 'end var', np.var(acts))
    tol_var = 0.1
    t_max = 10
    t_i = 0
    while True:
        acts1 = model(X)
        var = np.var(acts1)
        print('var', t_i, var)
        if abs(var - 1.0) < tol_var or t_i > t_max:
            break
        model.W /= model.ops.xp.sqrt(var)
        t_i += 1
    acts = model(X)
    print(model.name, 'end var', np.var(acts))
