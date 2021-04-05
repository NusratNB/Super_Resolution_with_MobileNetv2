from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf


def depth_to_space(inp, scale):
    return tf.nn.depth_to_space(inp, scale)


def relu6(x):
    return ReLU(max_value=6.0)(x)


def _upscale(inp, inp_filter, scale):
    x = depth_to_space(inp, scale)
    x = Conv2D(inp_filter, (1, 1), strides=(1, 1), padding='valid')(x)
    return x


def block(inp, out_filters, exp_ratio):
    channel = K.image_data_format()
    if channel == 'channel_last':
        channel_axis = -1
    else:
        channel_axis = 1
    inp_channel = K.int_shape(inp)[channel_axis]
    exp_filter = inp_channel * exp_ratio
    x = Conv2D(exp_filter, (1, 1), padding='valid')(inp)
    x = BatchNormalization()(x)
    x = relu6(x)
    x = DepthwiseConv2D((3, 3), padding='valid')(x)
    x = relu6(x)
    x = Conv2D(out_filters, (1, 1), padding='valid')(x)
    return x


def blocks(inp, out_filt, t, n):
    x = block(inp, out_filt, t)
    for i in range(1, n):
        x = block(x, out_filt, t)
    return x


#
def model(inp_shape):
    inputs = Input(shape=inp_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', name='input_layer')(inputs)
    x = blocks(x, 16, 1, 1)
    x = blocks(x, 32, 6, 1)
    x = blocks(x, 64, 6, 1)
    x = _upscale(x, 32, 4)
    x = _upscale(x, 48, 4)
    x = depth_to_space(x, 4)
    sr_model = Model(inputs, x)
    sr_model.summary()
    return sr_model