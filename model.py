from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def depth_to_space(inp, scale):
    return tf.nn.depth_to_space(inp, scale)


def linear(x):
    return tf.keras.activations.linear(x)


def relu6(x):
    return ReLU(max_value=6.0)(x)


def _upscale(inp, inp_filter, scale):
    x = depth_to_space(inp, scale)
    x = Conv2D(inp_filter, (1, 1), strides=(1, 1), padding='same')(x)
    return x


def block(inp, out_filters, exp_ratio):
    channel = K.image_data_format()
    if channel == 'channel_last':
        channel_axis = -1
    else:
        channel_axis = 1
    inp_channel = K.int_shape(inp)[channel_axis]
    exp_filter = inp_channel * exp_ratio
    x = Conv2D(exp_filter, (1, 1), padding='same')(inp)
    x = BatchNormalization()(x)
    x = relu6(x)
    x = DepthwiseConv2D((3, 3), padding='same', strides=(2, 2))(x)
    x = relu6(x)
    x = Conv2D(out_filters, (1, 1), padding='same')(x)
    x = linear(x)
    return x


def blocks(inp, out_filt, t, n):
    x = block(inp, out_filt, t)
    for i in range(1, n):
        x = block(x, out_filt, t)
    return x


def nn(inp_shape):
    inputs = Input(shape=inp_shape)
    x = Conv2D(8, (3, 3), strides=(2, 2), padding='same', name='input_layer')(inputs)
    x = blocks(x, 16, 1, 1)
    x = blocks(x, 32, 6, 1)
    x = blocks(x, 64, 6, 1)
    x = _upscale(x, 32, 2)
    x = _upscale(x, 48, 2)
    x = depth_to_space(x, 4)
    sr_model = Model(inputs, x)
    sr_model.summary()
    return sr_model
