
import math

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_train.utils.data_format import get_tf_data_format, get_channel_index, get_image_size, get_tensor_shape


def upsample_interpolation_function(inputs, factors, interpolation_function, support, name, data_format, padding):
    with tf.variable_scope(name):
        dim = len(factors)
        kernel = get_filler_kernel(interpolation_function, support, factors)
        kernel = kernel.reshape(kernel.shape + (1, 1))
        padding = padding.upper()

        # set padding and cropping parameters
        cropping = False
        if padding == 'VALID_CROPPED':
            cropping = True
            padding = 'VALID'

        # calculate convolution parameters
        input_size = get_image_size(inputs, data_format)
        data_format_tf = get_tf_data_format(inputs, data_format)
        inputs_shape = inputs.get_shape().as_list()
        channel_axis = get_channel_index(inputs, data_format)
        num_inputs = inputs_shape[channel_axis]

        # calculate output_size dependent on padding parameter
        if padding == 'SAME':
            output_size = [input_size[i] * factors[i] for i in range(dim)]
        else:
            output_size = [input_size[i] * factors[i] + kernel.shape[i] - factors[i] for i in range(dim)]
        output_shape = get_tensor_shape(batch_size=inputs_shape[0], channel_size=1, image_size=output_size, data_format=data_format)

        # strides are the scaling factors
        strides = get_tensor_shape(batch_size=1, channel_size=1, image_size=factors, data_format=data_format)

        # actual tensorflow operations - channelwise!
        split_inputs = tf.split(inputs, num_inputs, axis=channel_axis, name='split')
        output_list = []
        for i in range(len(split_inputs)):
            if dim == 2:
                current_output = tf.nn.conv2d_transpose(split_inputs[i], kernel, output_shape, strides, data_format=data_format_tf, name='conv' + str(i), padding=padding)
            else: # dim == 3
                current_output = tf.nn.conv3d_transpose(split_inputs[i], kernel, output_shape, strides, data_format=data_format_tf, name='conv' + str(i), padding=padding)
            output_list.append(current_output)
        outputs = tf.concat(output_list, axis=channel_axis, name='concat')

        # make a final cropping, if specified
        if cropping:
            image_paddings = [int((kernel.shape[i] - factors[i]) / 2) for i in range(dim)]
            paddings = get_tensor_shape(batch_size=0, channel_size=0, image_size=image_paddings, data_format=data_format)
            output_size_cropped = [input_size[i] * factors[i] for i in range(dim)]
            outputs = tf.slice(outputs, paddings, [inputs_shape[0], inputs_shape[1]] + output_size_cropped)

        return tf.identity(outputs, name='output')


def get_filler_kernel(f, support, factors):
    dim = len(factors)
    kernel_size = [2 * support * factor - factor % 2 for factor in factors]
    centers = [(2 * support * factor - 1 - factor % 2) * 0.5 for factor in factors]
    vec_f = np.vectorize(f)
    if dim == 2:
        y = np.array(range(kernel_size[0]), dtype=np.float32)
        x = np.array(range(kernel_size[1]), dtype=np.float32)
        f_y = vec_f((y - centers[0]) / factors[0])
        f_x = vec_f((x - centers[1]) / factors[1])
        f_y = f_y.reshape((-1, 1))
        f_x = f_x.reshape((1, -1))
        weights = f_y * f_x
    else: # dim == 3
        z = np.array(range(kernel_size[0]), dtype=np.float32)
        y = np.array(range(kernel_size[1]), dtype=np.float32)
        x = np.array(range(kernel_size[2]), dtype=np.float32)
        f_z = vec_f((z - centers[0]) / factors[0])
        f_y = vec_f((y - centers[1]) / factors[1])
        f_x = vec_f((x - centers[2]) / factors[2])
        f_z = f_z.reshape((-1, 1, 1))
        f_y = f_y.reshape((1, -1, 1))
        f_x = f_x.reshape((1, 1, -1))
        weights = f_z * f_y * f_x
    return weights.astype(np.float32)


def f_linear(x):
    return 1 - abs(x) if abs(x) <= 1 else 0


def f_cubic(x):
    A = -0.5
    if abs(x) <= 1:
        return (A + 2) * abs(x) ** 3 - (A + 3) * abs(x) ** 2 + 1
    if abs(x) < 2:
        return A * abs(x) ** 3 - 5 * A * abs(x) ** 2 + 8 * A * abs(x) - 4 * A
    return 0


def f_lanczos(x, A):
    if x == 0:
        return 1
    if abs(x) < A:
        return (A * math.sin(math.pi * x) * math.sin(math.pi * x / A)) / (math.pi * math.pi * x * x)
    return 0


def upsample_linear(inputs, factors, name, data_format='channels_first', padding='same'):
    return upsample_interpolation_function(inputs=inputs,
                                           factors=factors,
                                           interpolation_function=f_linear,
                                           support=1,
                                           name=name,
                                           data_format=data_format,
                                           padding=padding)


def upsample_cubic(inputs, factors, name, data_format='channels_first', padding='same'):
    return upsample_interpolation_function(inputs=inputs,
                                           factors=factors,
                                           interpolation_function=f_cubic,
                                           support=2,
                                           name=name,
                                           data_format=data_format,
                                           padding=padding)


def upsample_lanczos(inputs, factors, name, order=4, data_format='channels_first', padding='same'):
    interpolation_function = lambda x: f_lanczos(x, order)
    return upsample_interpolation_function(inputs=inputs,
                                           factors=factors,
                                           interpolation_function=interpolation_function,
                                           support=order,
                                           name=name,
                                           data_format=data_format,
                                           padding=padding)


def check_2d(inputs, factors):
    assert inputs.shape.ndims == 4, 'Invalid input tensor shape'
    assert len(factors) == 2, 'Invalid number of factors'


def check_3d(inputs, factors):
    assert inputs.shape.ndims == 5, 'Invalid input tensor shape'
    assert len(factors) == 3, 'Invalid number of factors'


def upsample2d_linear(inputs, factors, name, data_format='channels_first', padding='same'):
    check_2d(inputs, factors)
    return upsample_linear(inputs, factors, name, data_format, padding)


def upsample2d_cubic(inputs, factors, name, data_format='channels_first', padding='same'):
    check_2d(inputs, factors)
    return upsample_cubic(inputs, factors, name, data_format, padding)


def upsample2d_lanczos(inputs, factors, name, order=4, data_format='channels_first', padding='same'):
    check_2d(inputs, factors)
    return upsample_lanczos(inputs, factors, name, order, data_format, padding)


def upsample3d_linear(inputs, factors, name, data_format='channels_first', padding='same'):
    check_3d(inputs, factors)
    return upsample_linear(inputs, factors, name, data_format, padding)


def upsample3d_cubic(inputs, factors, name, data_format='channels_first', padding='same'):
    check_3d(inputs, factors)
    return upsample_cubic(inputs, factors, name, data_format, padding)


def upsample3d_lanczos(inputs, factors, name, order=4, data_format='channels_first', padding='same'):
    check_3d(inputs, factors)
    return upsample_lanczos(inputs, factors, name, order, data_format, padding)