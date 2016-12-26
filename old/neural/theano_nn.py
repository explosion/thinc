import theano
import theano.tensor as T



import numpy
from collections import OrderedDict, defaultdict


theano.config.floatX = 'float32'
floatX = theano.config.floatX


def th_share(w, name=''):
    return theano.shared(value=w, borrow=True, name=name)


class AvgParam(object):
    def __init__(self, numpy_data, name='?', wrapper=th_share):
        self.curr = wrapper(numpy_data, name=name+'_curr')
        self.avg = self.curr
        self.avg = wrapper(numpy_data.copy(), name=name+'_avg')
        self.step = wrapper(numpy.zeros(numpy_data.shape, numpy_data.dtype),
                            name=name+'_step')

    def updates(self, cost, timestep, momentum=0.9, eta=0.001):
        #return [(self.curr, self.curr - (eta * T.grad(cost, self.curr)))]
        step = (momentum * self.step) - T.grad(cost, self.curr)
        curr = self.curr + (eta * step)
        alpha = (1 / timestep).clip(0.001, 0.9).astype(floatX)
        avg = ((1 - alpha) * self.avg) + (alpha * curr)
        return [(self.curr, curr), (self.step, step), (self.avg, avg)]


def feed_layer(activation, weights, bias, input_):
    return activation(T.dot(input_, weights) + bias)


def L2(L2_reg, *weights):
    return L2_reg * sum((w ** 2).sum() for w in weights)


def L1(L1_reg, *weights):
    return L1_reg * sum(abs(w).sum() for w in weights)


def relu(x):
    return x * (x > 0)


def _init_weights(n_in, n_out):
    rng = numpy.random.RandomState(1234)
    weights = numpy.asarray(
        numpy.random.normal(
            loc=0.0,
            scale=0.0001,
            size=(n_in, n_out)),
        dtype=theano.config.floatX
    )
    bias = 0.2 * numpy.ones((n_out,), dtype=theano.config.floatX)
    return [AvgParam(weights, name='W'), AvgParam(bias, name='b')]


def compile_theano_model(n_classes, n_hidden, n_in, L1_reg, L2_reg):
    costs = T.ivector('costs') 
    x = T.vector('x') 
    timestep = theano.shared(1)
    eta = T.scalar('eta').astype(floatX)

    maxent_W, maxent_b = _init_weights(n_hidden, n_classes)
    hidden_W, hidden_b = _init_weights(n_in, n_hidden)

    # Feed the inputs forward through the network
    p_y_given_x = feed_layer(
                    T.nnet.softmax,
                    maxent_W.curr,
                    maxent_b.curr,
                      feed_layer(
                        relu,
                        hidden_W.curr,
                        hidden_b.curr,
                        x))

    stabilizer = 1e-8
    cost = (
        -T.log(T.sum((p_y_given_x[0] + stabilizer) * T.eq(costs, 0)))
        + L1(L1_reg, hidden_W.curr, hidden_b.curr, x)
        + L2(L2_reg, hidden_W.curr, hidden_b.curr, x)
    )

    debug = theano.function(
        name='debug',
        inputs=[x, costs],
        outputs=[p_y_given_x, T.eq(costs, 0), p_y_given_x[0] * T.eq(costs, 0)],
    )

    train_model = theano.function(
        name='train_model',
        inputs=[x, costs, eta],
        outputs=[p_y_given_x[0], T.grad(cost, x), T.argmax(p_y_given_x, axis=1)],
        updates=(
            [(timestep, timestep + 1)] + 
             maxent_W.updates(cost, timestep, eta=eta) + 
             maxent_b.updates(cost, timestep, eta=eta) +
             hidden_W.updates(cost, timestep, eta=eta) +
             hidden_b.updates(cost, timestep, eta=eta)
        )
    )

    evaluate_model = theano.function(
        name='evaluate_model',
        inputs=[x],
        outputs=[
            feed_layer(
              T.nnet.softmax,
              maxent_W.avg,
              maxent_b.avg,
              feed_layer(
                relu,
                hidden_W.avg,
                hidden_b.avg,
                x
              )
            )[0]
        ]
    )
    return debug, train_model, evaluate_model
