# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free

from collections import defaultdict
import numpy
from cupy import get_array_module
import cupy.linalg

from ..typedefs cimport weight_t
from ..linalg cimport Mat, MatMat, MatVec, VecVec, Vec, sqrt


def linear_decay(rate, decay, nr_upd):
    return rate * 1./(1. + decay * nr_upd)


def gpu_clip_gradient(gradient, threshold):
    grad_norm = cupy.linalg.norm(gradient)
    if grad_norm >= threshold:
        gradient *= threshold / grad_norm

@cython.cdivision(True)
cdef void cpu_update_averages(weight_t* ema,
        const weight_t* weights, int nr_weight, weight_t t, weight_t max_decay) nogil:
    cdef weight_t decay = (1.0 + t) / (10.0 + t)
    if decay > max_decay:
        decay = max_decay
    for i in range(nr_weight):
        ema[i] -= (1-decay) * (ema[i] - weights[i])


def gpu_update_averages(ema, weights, t, max_decay):
    cdef weight_t decay = (1.0 + t) / (10.0 + t)
    if decay > max_decay:
        decay = max_decay
    ema -= (1-decay) * (ema - weights)


def update_averages(averages, key, weights,
        nr_upd, max_decay=0.9999):
    xp = get_array_module(weights)
    if key not in averages:
        averages[key] = xp.zeros((weights.size,), dtype='float32')
    if xp is cupy:
        return gpu_update_averages(averages[key], weights, nr_upd, max_decay)
    cdef weight_t[::1] avg = averages[key]
    cdef weight_t[::1] W = weights
    cpu_update_averages(&avg[0],
        &W[0], avg.shape[0], nr_upd, max_decay)


def cpu_clip_gradient(weight_t[::1] gradient, weight_t threshold):
    grad_norm = Vec.norm(&gradient[0], gradient.shape[0])
    if grad_norm >= threshold:
        Vec.mul_i(&gradient[0], threshold / grad_norm, gradient.shape[0])


def add_gradient_noise(float[::1] gradient, weight_t noise_level,
        weight_t timestep):
    variance = noise_level / ((1 + timestep) ** 0.55)
    if variance >= 0.000001:
        gradient += numpy.asarray(
                       numpy.random.normal(scale=variance, loc=0., size=len(gradient)),
                       dtype='float32')


class SGD(object):
    def __init__(self, ops, lr, momentum=0.0, decay=0.0, **settings):
        self.ops = ops
        self.alpha = lr
        self.mu = momentum
        self.decay = decay
        self.momentums = {}
        self.averages = {} if settings.get('averages', True) else None
        self.nr_update = defaultdict(int)

    @property
    def nr_iter(self):
        if not self.nr_update:
            return 0
        return max(self.nr_update.values())

    def __call__(self, weights, gradient, key=None, lr_scale=1.):
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        lr = self.lr(nr_upd)
        lr *= lr_scale
        gpu_clip_gradient(gradient, 10.)
        if key is None or self.mu == 0.0:
            weights -= lr * gradient
            gradient.fill(0)
        else:
            if key not in self.momentums:
                self.momentums[key] = self.ops.allocate(weights.size)
            momentum = self.momentums[key]
            momentum *= self.mu
            momentum += gradient * lr
            weights -= momentum
            gradient.fill(0)
        if self.averages is not None:
            update_averages(self.averages, key, weights, nr_upd)

    def lr(self, nr_upd):
        return linear_decay(self.alpha, self.decay, nr_upd)


class Adam(SGD):
    def __init__(self, ops, lr, beta1=0.90, beta2=0.999, eps=1e-08, decay=0.0):
        self.ops = ops
        self.mom1 = {}
        self.mom2 = {}
        self.averages = {}
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self.alpha = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.decay = decay
        self.d = 1.
        self.f = 0.

    def lr(self, nr_upd):
        alpha = linear_decay(self.alpha, self.decay, nr_upd)
        fix1 = 1.- (self.b1 ** nr_upd)
        fix2 = 1.- (self.b2 ** nr_upd)
        return alpha * numpy.sqrt(fix2) / fix1
    
    def __call__(self, weights, gradient, lr_scale=1., 
            key=None):
        assert key is not None
        assert len(gradient) >= 1
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.size)
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        gpu_clip_gradient(gradient, 10.)

        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        cdef weight_t lr = self.lr(nr_upd) * lr_scale
        cdef weight_t b1 = self.b1
        cdef weight_t b2 = self.b1
        cdef weight_t eps = self.eps
        
        gpu_adam(weights, gradient, mom1, mom2, b1, b2, eps, lr)
        gradient.fill(0)
        #_adam(&weights[0], &gradient[0], &mom1[0], &mom2[0],
        #    weights.shape[0], b1, b2, eps, lr)
        
        if self.averages is not None:
            update_averages(self.averages, key, weights, nr_upd)

    def set_loss(self, loss):
        pass


@cython.cdivision(True)
cdef void _adam(
    weight_t* weights, weight_t* gradient, weight_t* mom1, weight_t* mom2, 
        int nr_weight, weight_t beta1, weight_t beta2, weight_t eps,
        weight_t learn_rate) nogil:
    Vec.mul_i(mom1,
        beta1, nr_weight)
    VecVec.add_i(mom1,
        gradient, 1-beta1, nr_weight)

    for i in range(nr_weight):
        gradient[i] *= gradient[i] * (1-beta2)
    for i in range(nr_weight):
        mom2[i] = (beta2 * mom2[i]) + gradient[i]
    #for i in range(nr_weight):
    #    mom2[i] = (beta2 * mom2[i]) + ((1-beta2) * gradient[i] * gradient[i])
    # Here we assume this is calculated by the caller.
    #cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t)
    for i in range(nr_weight):
        weights[i] -= learn_rate * (mom1[i] / (sqrt(mom2[i]) + eps))
    memset(gradient, 0, sizeof(gradient[0]) * nr_weight)


def gpu_adam(weights, gradient, mom1, mom2, beta1, beta2, eps, learn_rate):
    mom1 *= beta1
    mom2 *= beta2
    mom1 += gradient * (1.-beta1)
    mom2 += gradient * gradient * (1.-beta2)
    # Here we assume learn rate is calculated by the caller.
    # cdef weight_t a_t = learn_rate * sqrt(1-beta2**hp.t) / (1-beta1**hp.t);
    weights -= learn_rate * (mom1 / (cupy.sqrt(mom2) + eps))

class Eve(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.b3 = 0.999
        self.lower_threshold = 0.1
        self.upper_threshold = 10
        self.d = 1.
        self.f = None

    def __getattr__(self, attr):
        return getattr(self.optimizer, attr)

    def __call__(self, weights, gradient, key=None):
        return self.optimizer(weights, gradient, key=key,
            lr_scale=self.d)

    def set_loss(self, loss):
        if self.f is None:
            self.f = loss
            return
        old_f = self.f
        d = self.d
        c = self._get_c(loss, old_f)
        new_f = c * loss
        r = abs(new_f - old_f) / min(new_f, old_f)
        new_d = d + (1 - self.b3) * (r - d)
        self.d = new_d
        self.f = new_f

    def _get_c(self, loss, old_f):
        if loss < old_f:
            delta = self.lower_threshold + 1.
            Delta = self.upper_threshold + 1.
        else:
            delta = 1. / (self.upper_threshold + 1.)
            Delta = 1. / (self.lower_threshold + 1.)
        return min(max(delta, loss / old_f), delta)
