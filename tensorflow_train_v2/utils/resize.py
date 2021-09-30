"""
Functions for resizing.
"""

import tensorflow as tf
from tensorflow_train_v2.utils.data_format import get_batch_channel_image_size, get_image_size


def get_tf_interpolator(interpolator):
    """
    Return a tf resize method object for the given string.
    :param interpolator: Interpolator type as string.
                         'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR
                         'linear': tf.image.ResizeMethod.BILINEAR
                         'cubic': tf.image.ResizeMethod.BICUBIC
                         'area': tf.image.ResizeMethod.AREA
                         'lanczos': tf.image.ResizeMethod.LANCZOS3
    :return: The sitk interpolator object.
    """
    if interpolator == 'nearest':
        return tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif interpolator == 'linear':
        return tf.image.ResizeMethod.BILINEAR
    elif interpolator == 'cubic':
        return tf.image.ResizeMethod.BICUBIC
    elif interpolator == 'area':
        return tf.image.ResizeMethod.AREA
    elif interpolator == 'lanczos':
        return tf.image.ResizeMethod.LANCZOS3
    else:
        raise Exception(f'invalid interpolator type {interpolator}')


def resize(inputs, factors=None, output_size=None, interpolator='linear', name=None, data_format='channels_first'):
    """
    Resizes a image tensor to either a given size of a factor.
    :param inputs: 4d or 5d tensor.
    :param output_size: Output size.
    :param interpolator: The interpolator. See get_tf_interpolator.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    inputs = tf.convert_to_tensor(inputs)
    if inputs.shape.ndims == 4:
        return resize2d(inputs, factors=factors, output_size=output_size, interpolator=interpolator, name=name, data_format=data_format)
    elif inputs.shape.ndims == 5:
        return resize3d(inputs, factors=factors, output_size=output_size, interpolator=interpolator, name=name, data_format=data_format)
    else:
        raise ValueError(f'Only image tensors with ndims == 4 or 5 are supported, is {inputs.shape.ndims}')


def resize3d(inputs, factors=None, output_size=None, interpolator='linear', name=None, data_format='channels_first'):
    """
    Resizes a 3D input volume to either a given size of a factor.
    :param inputs: 5D tensor.
    :param output_size: Output size.
    :param interpolator: The interpolator. See get_tf_interpolator.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    with tf.name_scope(name or 'resize3d'):
        num_batches, num_channels, image_size = get_batch_channel_image_size(inputs, data_format=data_format, as_tensor=True)
        depth, height, width = image_size[0], image_size[1], image_size[2]
        dtype = inputs.dtype
        interpolator_tf = get_tf_interpolator(interpolator)
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(inputs, [0, 2, 3, 4, 1])
        else:
            inputs_channels_last = inputs

        if output_size is None:
            output_depth, output_height, output_width = [tf.cast(tf.cast(s, tf.float32) * f, tf.int32) for s, f in zip([depth, height, width], factors)]
        else:
            output_depth, output_height, output_width = output_size[0], output_size[1], output_size[2]

        # resize y-z
        squeeze_b_x = tf.reshape(inputs_channels_last, [-1, height, width, num_channels])
        resize_b_x = tf.cast(tf.image.resize(squeeze_b_x, [output_height, output_width], method=interpolator_tf), dtype=dtype)
        resume_b_x = tf.reshape(resize_b_x, [num_batches, depth, output_height, output_width, num_channels])

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(reoriented, [-1, output_height, depth, num_channels])
        resize_b_z = tf.cast(tf.image.resize(squeeze_b_z, [output_height, output_depth], method=interpolator_tf), dtype=dtype)
        resume_b_z = tf.reshape(resize_b_z, [num_batches, output_width, output_height, output_depth, num_channels])

        if data_format == 'channels_first':
            output = tf.transpose(resume_b_z, [0, 4, 3, 2, 1])
        else:
            output = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        return output


def resize2d(inputs, output_size=None, factors=None, interpolator='linear', name=None, data_format='channels_first'):
    """
    Resizes a 2D input volume to either a given size of a factor.
    :param inputs: 4D tensor.
    :param output_size: Output size.
    :param interpolator: The interpolator. See get_tf_interpolator.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    with tf.name_scope(name or 'resize2d'):
        input_size = get_image_size(inputs, data_format=data_format, as_tensor=True)
        dtype = inputs.dtype
        interpolator_tf = get_tf_interpolator(interpolator)
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(inputs, [0, 2, 3, 1])
        else:
            inputs_channels_last = inputs
        if output_size is None:
            output_size = [int(s * f) for s, f in zip(input_size, factors)]
        upsampled_channels_last = tf.cast(tf.image.resize(inputs_channels_last, output_size, method=interpolator_tf), dtype=dtype)
        if data_format == 'channels_first':
            outputs = tf.transpose(upsampled_channels_last, [0, 3, 1, 2])
        else:
            outputs = upsampled_channels_last
    return outputs
