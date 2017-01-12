from .model import Model


class ScaleShift(Model):
    def begin_update(self, input__BI):
        def finish_update(gradient__BI):
            self.d.b += gradient__BI.sum(axis=0)
            for i in range(gradient__BI.shape[0]):
                self.d.G += gradient__BI[i] * input__BI[i]
            return gradient__BI * self.d.G
        return input__BI * self.d.G + self.w.b, finish_update


class BatchNormalization(Model):
    def predict(self, X):
        N, mu, var = _get_moments(self.ops, X)
        return _forward(self.ops, X, mu, var)
 
    def begin_update(self, X):
        N, mu, var = _get_moments(self.ops, X)
        Xhat = _forward(self.ops, X, mu, var)

        def finish_update(dy):
            dist, sum_dy, sum_dy_dist = _get_d_moments(self.ops, dy, X, mu)
            d_xhat = N * dy - sum_dy - dist * var**(-1.) * sum_dy_dist
            d_xhat *= var ** (-1. / 2)
            d_xhat /= N
            return d_xhat
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


#    @property
#    def G(self):
#        return self.params.get('G-%s' % self.name)
#
#    @property
#    def b(self):
#        return self.params.get('b-%s' % self.name)
#
#    @property
#    def d_G(self):
#        return self.params.get('d_G-%s' % self.name, require=True)
#
#    @property
#    def d_b(self):
#        return self.params.get('d_b-%s' % self.name, require=True)
#
#    @property
#    def describe_params(self):
#        '''
#        Yields (name, shape, initializer) triples describing the weights directly
#        owned by the layer.
#        '''
#        def init(G, **kwargs):
#            G += 1
#        yield 'G-%s' % self.name, (self.nr_out,), init
#        yield 'b-%s' % self.name, (self.nr_out,), None
#
#
