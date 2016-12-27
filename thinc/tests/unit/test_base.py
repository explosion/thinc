import pytest

from ...base import Model

class MockOps(object):
    pass


class Subclass(Model):
    @property
    def shape(self):
        return self._shape

    @property
    def nr_out(self):
        return self.shape[0]

    @property
    def nr_in(self):
        return self.shape[1]

    def setup(self, *shape, **kwargs):
        self._shape = tuple(shape)

    def predict_batch(self, X):
        return ['label' for _ in range(len(X))]

    def predict_one(self, x):
        return 'label'

    def check_shape(self, x, is_batch):
        pass

    def is_batch(self, x):
        return True if isinstance(x, list) else False


def test_specify_shape():
    sub = Subclass(2, 2, ops=MockOps())
    assert sub.shape == (2,2)
    assert sub.nr_out == 2
    assert sub.nr_in == 2


def test_predict():
    sub = Subclass(2, 2, ops=MockOps())
    assert sub('question') == 'label'
    assert sub(['q1', 'w2']) == ['label', 'label']

    for result in sub.pipe(['q1', 'w2']):
        assert result == 'label'


def test_begin_update():
    sub = Subclass(2, 2, ops=MockOps())
    with pytest.raises(NotImplementedError):
        sub.begin_update('question', dropout=0.0)
