
import numpy as np
import tensorflow as tf

from tensorflow_train_v2.utils.data_format import get_image_axes
from tensorflow_train_v2.utils.tensorflow_util import shift_with_padding


def upsample_axis_with_f(x, factor, axis, f, support):
    """
    Upsample axis in x by an integer factor and apply the given interpolation function that has the given support.
    :param x: Tensor.
    :param factor: The upsampling factor.
    :param axis: The axis to upsample.
    :param f: The function to apply.
    :param support: The support of the function to apply.
    :return: The upsampled Tensor.
    """
    auxiliary_axis = axis + 1
    dim = len(x.shape)
    dtype = x.dtype

    # prepare shifted x with repeat border handling
    xs = []
    for shift in range(-support, support + 1):
        xs.append(shift_with_padding(x, shift, axis, padding='repeat'))

    # calculate multiplicative factors and stack arrays
    x_rep_list = []
    for i in range(factor):
        # position_on_line is where the center offset is located on the function f
        position_on_line = (1 - factor + i * 2) / (factor * 2)
        current_xs = []
        for current_x, offset in zip(xs, range(-support, support + 1)):
            # calculate the individual multiplicative factors from offsets based on the support of f
            prod_factor = f(position_on_line + offset)
            if prod_factor != 0:
                current_xs.append(current_x * prod_factor)
        x_rep_list.append(tf.reduce_sum(current_xs, axis=0))
    x_rep = tf.stack(x_rep_list, axis=auxiliary_axis)

    # merge
    reps = np.ones(dim)
    reps[axis] = factor
    reps = tf.constant(reps, dtype='int32')
    x_shape = tf.shape(x)
    x_shape *= reps
    x_rep = tf.reshape(x_rep, x_shape)

    # fix shape representation
    x_shape = x.shape.as_list()
    if not x_shape[axis] is None:
        x_shape[axis] *= factor
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)
    return x_rep


def f_linear(x):
    """
    Linear interpolation function.
    :param x: Distance to center (0).
    :return: Function value.
    """
    return 1 - abs(x) if abs(x) <= 1 else 0


def f_cubic(x):
    """
    Cubic interpolation function.
    :param x: Distance to center (0).
    :return: Function value.
    """
    A = -0.5
    if abs(x) <= 1:
        return (A + 2) * abs(x) ** 3 - (A + 3) * abs(x) ** 2 + 1
    if abs(x) < 2:
        return A * abs(x) ** 3 - 5 * A * abs(x) ** 2 + 8 * A * abs(x) - 4 * A
    return 0


def f_lanczos(x, A):
    """
    Lanczos interpolation function for given order A.
    :param x: Distance to center (0).
    :param A: Order.
    :return: Function value.
    """
    if x == 0:
        return 1
    if abs(x) < A:
        return (A * np.sin(np.pi * x) * np.sin(np.pi * x / A)) / (np.pi * np.pi * x * x)
    return 0


def upsample_axis_linear(x, factor, axis):
    """
    Upsample axis in x by an integer factor and apply linear interpolation.
    :param x: Tensor.
    :param factor: The upsampling factor.
    :param axis: The axis to upsample.
    :return: The upsampled Tensor.
    """
    return upsample_axis_with_f(x, factor, axis, f_linear, 1)


def upsample_axis_cubic(x, factor, axis):
    """
    Upsample axis in x by an integer factor and apply cubic interpolation.
    :param x: Tensor.
    :param factor: The upsampling factor.
    :param axis: The axis to upsample.
    :return: The upsampled Tensor.
    """
    return upsample_axis_with_f(x, factor, axis, f_cubic, 2)


def upsample_axis_lanczos(x, factor, axis):
    """
    Upsample axis in x by an integer factor and apply lanczos interpolation.
    :param x: Tensor.
    :param factor: The upsampling factor.
    :param axis: The axis to upsample.
    :return: The upsampled Tensor.
    """
    order = 4
    return upsample_axis_with_f(x, factor, axis, lambda x: f_lanczos(x, order), order)


def upsample_axis(inputs, factors, upsample_axis_function, data_format='channels_first'):
    """
    Upsample an image or volume tensor for the given integer factors with the given upsample axis function.
    :param inputs: The tensor input.
    :param factors: The list of integer factors.
    :param upsample_axis_function: The function to apply.
    :param data_format: The data_format.
    :return: The upsampled tensor.
    """
    image_axes = get_image_axes(inputs, data_format)
    outputs = inputs
    for factor, axis in zip(factors, image_axes):
        outputs = upsample_axis_function(outputs, factor, axis)
    return outputs


def upsample_linear(inputs, factors, data_format='channels_first'):
    """
    Upsample an image or volume tensor for the given integer factors with linear interpolation.
    :param inputs: The tensor input.
    :param factors: The list of integer factors.
    :param data_format: The data_format.
    :return: The upsampled tensor.
    """
    return upsample_axis(inputs, factors, upsample_axis_linear, data_format)


def upsample_cubic(inputs, factors, data_format='channels_first'):
    """
    Upsample an image or volume tensor for the given integer factors with cubic interpolation.
    :param inputs: The tensor input.
    :param factors: The list of integer factors.
    :param data_format: The data_format.
    :return: The upsampled tensor.
    """
    return upsample_axis(inputs, factors, upsample_axis_cubic, data_format)


def upsample_lanczos(inputs, factors, data_format='channels_first'):
    """
    Upsample an image or volume tensor for the given integer factors with lanczos interpolation.
    :param inputs: The tensor input.
    :param factors: The list of integer factors.
    :param data_format: The data_format.
    :return: The upsampled tensor.
    """
    return upsample_axis(inputs, factors, upsample_axis_lanczos, data_format)
