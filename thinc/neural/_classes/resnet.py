from .model import Model
from .batchnorm import BatchNorm
from ...api import layerize
from .affine import Affine

import cytoolz as toolz


def Residual(layer):
	def forward(X, drop=0.):
		y, bp_y = layer.begin_update(X, drop=drop)
		output = X+y
		def backward(d_output, sgd=None):
			return d_output + bp_y(d_output, sgd)
		return output, backward
	model = layerize(forward)
	model._layers.append(layer)
	def on_data(self, X, y=None):
		for layer in self._layers:
			for hook in layer.on_data_hooks:
				hook(layer, X, y)
	model.on_data_hooks.append(on_data)
	return model
