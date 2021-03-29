from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, BatchNormalization
from tensorflow.keras.layers import ReLU, Input, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def relu6(x):
    return ReLU(max_value=6.0)(x)


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


def model(inp_shape):
    inputs = Input(shape=inp_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2))(inputs)
    x = blocks(x, 16, 1, 1)
    x = blocks(x, 24, 6, 2)
    x = blocks(x, 32, 6, 3)
    x = Conv2DTranspose(64, (3, 3))(x)
    x = Conv2DTranspose(48, (3, 3))(x)
    x = Conv2DTranspose(32, (3, 3))(x)
    x = Conv2DTranspose(24, (3, 3))(x)
    x = Conv2DTranspose(16, (3, 3))(x)
    x = Conv2DTranspose(16, (3, 3))(x)
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2))(x)
    sr_model = Model(inputs, x)
    return sr_model