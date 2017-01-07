from .model import Model


class ScaleShift(Model):
    name = 'bnscale'
    @property
    def G(self):
        return self.params.get('G-%s' % self.name)

    @property
    def b(self):
        return self.params.get('b-%s' % self.name)

    @property
    def d_G(self):
        return self.params.get('d_G-%s' % self.name, require=True)

    @property
    def d_b(self):
        return self.params.get('d_b-%s' % self.name, require=True)

    @property
    def describe_params(self):
        '''
        Yields (name, shape, initializer) triples describing the weights directly
        owned by the layer.
        '''
        def init(G, **kwargs):
            G += 1
        yield 'G-%s' % self.name, (self.nr_out,), init
        yield 'b-%s' % self.name, (self.nr_out,), None

    def begin_update(self, input_BI, dropout=0.0, **kwargs):
        def finish_update(gradient_BI, *args, **kwargs):
            d_b = self.d_b
            d_b += gradient_BI.sum(axis=0)
            d_G = self.d_G
            for i in range(gradient_BI.shape[0]):
                d_G += gradient_BI[i] * input_BI[i]
            return gradient_BI * self.G
        return input_BI * self.G + self.b, finish_update

    def __init__(self, nr_out, *args, **kwargs):
        self.nr_out = nr_out
        Model.__init__(self, *args, **kwargs)
 

class BatchNormalization(Model):
    def check_input(self, X, expect_batch=True):
        return True

    def predict_batch(self, X):
        N, mu, var = _get_moments(self.ops, X)
        return _forward(self.ops, X, mu, var)
 
    def begin_update(self, X, dropout=0.0):
        N, mu, var = _get_moments(self.ops, X)
        Xhat = _forward(self.ops, X, mu, var)

        def finish_update(dy, optimizer=None, **kwargs):
            assert len(X) == len(dy)
            dist, sum_dy, sum_dy_dist = _get_d_moments(self.ops, dy, X, mu)
            if hasattr(dy, 'shape'):
                d_xhat = N * dy - sum_dy - dist * var**(-1.) * sum_dy_dist
                d_xhat *= var ** (-1. / 2)
                d_xhat /= N
                return d_xhat
            else:
                seqs = (dy, sum_dy, dist, sum_dy_dist)
                output = []
                assert len(sum_dy) == len(dy)
                assert len(dist) == len(dy)
                assert len(sum_dy_dist) == len(dy)
                for dy_, sum_dy_, dist_, sum_dy_dist_ in zip(*seqs):
                    d_xhat = N * dy_ - sum_dy_ - dist_ * var**(-1.) * sum_dy_dist_
                    d_xhat *= var ** (-1. / 2)
                    d_xhat /= N
                    output.append(d_xhat)
                assert len(output) == len(dy), (len(output), len(dy))
                return output
        return Xhat, finish_update


def _get_moments(ops, X):
    if hasattr(X, 'shape') and len(X.shape) == 2:
        mu = X.mean(axis=0)
        var = X.var(axis=0) + 1e-8
        return X.shape[0], mu, var
    else:
        stacked = numpy.vstack(X)
        return stacked.shape[0], stacked.mean(axis=0), stacked.var(axis=0)


def _get_d_moments(ops, dy, X, mu):
    if hasattr(dy, 'shape'):
        dist = X-mu
        return dist, ops.xp.sum(dy, axis=0), ops.xp.sum(dy * dist, axis=0)
    else:
        sum_dy = [ops.xp.sum(seq, axis=0) for seq in dy]
        dist = [x-mu for x in X]
        sum_dy_dot_dist = [ops.xp.sum(seq * d, axis=0) for seq, d in zip(dy, dist)]
        return dist, sum_dy, sum_dy_dot_dist


def _forward(ops, X, mu, var):
    if hasattr(X, 'shape'):
        return (X-mu) * var ** (-1./2.)
    else:
        return [_forward(x, mu, var) for x in X]
