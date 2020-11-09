import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


class OverlapingLayer(Layer):

    def __init__(self, **kwargs):
        super(OverlapingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.b = self.add_weight(shape=(1,), initializer='zero', name='{}_b1'.format(self.name))
        self.w1 = self.add_weight(shape=(1,), initializer='one', name='{}_w1'.format(self.name))
        self.w2 = self.add_weight(shape=(1,), initializer='zero', name='{}_w2'.format(self.name))

        super(OverlapingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # return K.dot(x, self.kernel)
        r = []
        for i in range(x.shape[2]):
            t = x[:, :, :i]
            t = K.concatenate([t, x[:, :, :x.shape[2] - i]], axis=-1)
            r.append(K.expand_dims(x * self.w1[0] + (self.b[0] - t) * self.w2[0], axis=1))


        return K.concatenate(r, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[2])

class StereoConv(Layer):
    '''
    This layer do the same that the OverlapingLayer but using a *conv2d* operator.
    '''

    def __init__(self, **kwargs):
        super(StereoConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w1 = self.add_weight(shape=(1,), initializer='random_normal', name='{}_w1'.format(self.name))
        self.w2 = self.add_weight(shape=(1,), initializer='random_normal', name='{}_w2'.format(self.name))

        super(StereoConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        r = []

        for i in range(x.shape[-1]):
            kernel = K.concatenate((self.w1, np.zeros(i, dtype='float32'), self.w2))

            kernel = K.reshape(kernel, [1] + list(kernel.shape) + [1, 1])

            conv = K.conv2d(x, kernel, padding='same',
                            strides=(1, 1), dilation_rate=(1, 1),
                            data_format="channels_first")

            r.append(conv)

        return K.concatenate(r, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[3], input_shape[3], input_shape[2], input_shape[3])

