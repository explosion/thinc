import contextlib
from io import BytesIO
import numpy as np
from ..util import tensorflow2xp
import itertools

from .shim import Shim

try:
    import cupy
except ImportError:
    cupy = None

try:
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import h5py
    has_h5py = True
except ImportError:
    has_h5py = False


class TensorFlowShim(Shim):
    """Interface between a Tensorflow model and a Thinc Model. This container is
    *not* a Thinc Model subclass itself.

    reference for custom training:
    https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
    """

    def __call__(self, args, kwargs, is_train):
        if is_train:
            return self.begin_update(args, kwargs)
        else:
            return self.predict(args, kwargs), lambda args, kwargs: (args, kwargs)

    def predict(self, args, kwargs):
        tf.keras.backend.set_learning_phase(0)
        y_var = self._model(*args, **kwargs)
        tf.keras.backend.set_learning_phase(1)
        return y_var, lambda d_args, d_kwargs: d_args

    def begin_update(self, args, kwargs):
        tf.keras.backend.set_learning_phase(1)
        output = self._model(*args, **kwargs)

        def backprop(d_args, d_kwargs):
            with tf.GradientTape() as tape:
                self.tensorflow_grads = tape.gradient(*d_args,
                                                      self._model.trainable_variables)
            return self.tensorflow_grads
        return output, backprop

    def finish_update(self, optimizer):
        if not self._optimizer:
            self._optimizer = self._create_optimizer(optimizer)
        optimizer.apply_gradients(zip(self.tensorflow_grads["gradients"],
                                      self._model.trainable_variables))
        self._update_tensorflow_averages(optimizer)

    def _create_optimizer(self, sgd):
        if sgd.b1 != 0 and sgd.b2 != 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=sgd.alpha,
                                                 beta_1=sgd.b1,
                                                 beta_2=sgd.b2
                                                 )
        elif sgd.b2 == 0:
            optimizer = tf.keras.optimizers.SGD(learning_rate=sgd.alpha,
                                                momentum=sgd.b1
                                                )
        else:
            raise NotImplementedError
        return optimizer

    def _load_weights_from_state_dict(self, state_dict):
        for layer in self._model.layers:
            current_layer_weights = []
            for weight in layer.weights:
                current_layer_weights.append(state_dict[weight.name])
            layer.set_weights(current_layer_weights)

    # Create a state dict similar to PyTorch
    def _create_state_dict(self):
        # key as variable name and value as numpy arrays
        state_dict = {}
        for layer in self._model.layers:
            for weight in layer.weights:
                state_dict[weight.name] = weight.numpy()
        return state_dict

    @contextlib.contextmanager
    def use_params(self, params):
        key_prefix = f"tensorflow_{self.id}_"
        # state dict stores key as name and value as numpy array
        state_dict = {}
        for k, v in params.items():
            if hasattr(k, "startswith") and k.startswith(key_prefix):
                if cupy is None:
                    assert isinstance(v, np.ndarray)
                else:
                    if isinstance(v, cupy.core.core.ndarray):
                        v = cupy.asnumpy(v)
                    assert isinstance(v, np.ndarray)
                state_dict[k.replace(key_prefix, "")] = v
        if state_dict:
            backup = self._create_state_dict()
            self._load_weights_from_state_dict(state_dict)
            yield
            self._load_weights_from_state_dict(backup)
        else:
            yield

    def _update_tensorflow_averages(self, sgd, *, init_steps=1):
        if getattr(sgd, "averages", None) is None:
            return
        # Collect parameters if we don't have them
        layers = [l.weights for l in self._model.layers]
        layers = itertools.chain(*layers)
        for layer in layers:
            key = f"tensorflow_{self.id}_{layer.name}"
            sgd.nr_update[key] += 1
            xp_param = tensorflow2xp(layer)
            if key in sgd.averages:
                sgd.ops.update_averages(sgd.averages[key], xp_param, sgd.nr_update[key])
            else:
                sgd.averages[key] = xp_param.copy()
                sgd.nr_update[key] = init_steps

    def _clone_model(self):
        """similar to tf.keras.models.clone_model()
        But the tf.keras.models.clone_model changes the names of tf.Variables.
        This method even preserves that
        """
        state_dict = self._create_state_dict()
        model_json_config = self._model.to_json()
        tf.keras.backend.clear_session()
        self._model = tf.keras.models.model_from_json(model_json_config)
        self._load_weights_from_state_dict(state_dict)

    def to_gpu(self, device_num):
        assert isinstance(device_num, str), "device_num must be string like ''/GPU:0'"
        with tf.device(device_num):
            self._clone_model()

    def to_cpu(self):
        with tf.device("/CPU"):
            self._clone_model()

    def to_disk(self, path):
        self._model.save(path)

    def from_disk(self, path):
        if self.ops.device == "cpu":
            device = "CPU"
        else:
            device = tf.test.gpu_device_name()
        with tf.device(device):
            self._model = tf.keras.models.load_model(path)

    def to_bytes(self):
        filelike = BytesIO()
        with h5py.File(filelike, "w") as f:
            self._model.save(f, save_format="h5")
        return filelike.getvalue()

    def from_bytes(self, data):
        filelike = BytesIO(data)
        filelike.seek(0)
        if self.ops.device == "cpu":
            device = "CPU"
        else:
            device = tf.test.gpu_device_name()
        with h5py.File(filelike, "r") as f:
            with tf.device(device):
                self._model = tf.keras.models.load_model(f)
