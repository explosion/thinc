# coding: utf8
from __future__ import unicode_literals

from .model import Model


class FunctionLayer(Model):
    """Wrap functions into weightless Model instances, for use as network
    components."""

    def __init__(
        self,
        begin_update,
        predict=None,
        predict_one=None,
        nI=None,
        nO=None,
        *args,
        **kwargs
    ):
        self.begin_update = begin_update
        if predict is not None:
            self.predict = predict
        if predict_one is not None:
            self.predict_one = predict_one
        self.nI = nI
        self.nO = nO
        Model.__init__(self)
