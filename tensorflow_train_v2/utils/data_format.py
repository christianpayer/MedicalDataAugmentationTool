
import tensorflow as tf


def get_tf_data_format(inputs, data_format):
    """
    Return the tf data format string for the given 4D or 5D tensor.
    :param inputs: The tensor.
    :param data_format: 'channels_first' or 'channels_last'
    :return: 'NCHW' or 'NHWC' for 4D tensors; 'NCDHW', or /'NDHWC' for 5D tensors.
    """
    if inputs.shape.ndims == 4:
        return get_tf_data_format_2d(data_format)
    elif inputs.shape.ndims == 5:
        return get_tf_data_format_3d(data_format)
    else:
        raise Exception('unsupported inputs shape')


def get_tf_data_format_2d(data_format):
    """
    Return the tf data format string.
    :param data_format: 'channels_first' or 'channels_last'
    :return: 'NCHW' or 'NHWC'.
    """
    if data_format == 'channels_first':
        return 'NCHW'
    elif data_format == 'channels_last':
        return 'NHWC'
    else:
        raise Exception('unsupported data format')


def get_tf_data_format_3d(data_format):
    """
    Return the tf data format string.
    :param data_format: 'channels_first' or 'channels_last'
    :return: 'NCDHW' or 'NDHWC'.
    """
    if data_format == 'channels_first':
        return 'NCDHW'
    elif data_format == 'channels_last':
        return 'NDHWC'
    else:
        raise Exception('unsupported data format')


def get_channel_index(inputs, data_format):
    """
    Return the channel index for the given tensor and data_format.
    :param inputs: The tensor.
    :param data_format: 'channels_first' or 'channels_last'
    :return: The channel index.
    """
    if inputs.shape.ndims == 4:
        return get_channel_index_2d(data_format)
    elif inputs.shape.ndims == 5:
        return get_channel_index_3d(data_format)
    else:
        raise Exception('unsupported inputs shape')


def get_channel_index_2d(data_format):
    """
    Return the channel index for the given data_format.
    :param data_format: 'channels_first' or 'channels_last'
    :return: 1 or 3.
    """
    if data_format == 'channels_first':
        return 1
    elif data_format == 'channels_last':
        return 3
    else:
        raise Exception('unsupported data format')


def get_channel_index_3d(data_format):
    """
    Return the channel index for the given data_format.
    :param data_format: 'channels_first' or 'channels_last'
    :return: 1 or 4.
    """
    if data_format == 'channels_first':
        return 1
    elif data_format == 'channels_last':
        return 4
    else:
        raise Exception('unsupported data format')


def get_image_axes(inputs, data_format):
    """
    Return the channel index for the given tensor and data_format.
    :param inputs: The tensor.
    :param data_format: 'channels_first' or 'channels_last'
    :return: Tuple of the image axes.
    """
    if inputs.shape.ndims == 4:
        if data_format == 'channels_first':
            return 2, 3
        else:  # if data_format == 'channels_last':
            return 1, 2
    elif inputs.shape.ndims == 5:
        if data_format == 'channels_first':
            return 2, 3, 4
        else:  # if data_format == 'channels_last':
            return 1, 2, 3


def get_tensor_shape(inputs, as_tensor=False):
    """
    Return the shape of the tensor.
    :param inputs: The tensor.
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: Tuple of the image axes.
    """
    return tf.shape(inputs) if as_tensor else tuple(inputs.get_shape().as_list())


def get_image_size(inputs, data_format, as_tensor=False):
    """
    Return the image_size for the given inputs and data_format.
    :param inputs: The inputs.
    :param data_format: 'channels_first' or 'channels_last'
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: The image_size.
    """
    if inputs.shape.ndims == 4:
        return get_image_size_2d(inputs, data_format, as_tensor)
    elif inputs.shape.ndims == 5:
        return get_image_size_3d(inputs, data_format, as_tensor)
    else:
        raise Exception('unsupported inputs shape')


def get_image_size_2d(inputs, data_format, as_tensor=False):
    """
    Return the 2D image_size for the given inputs and data_format.
    :param inputs: The inputs.
    :param data_format: 'channels_first' or 'channels_last'
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: The image_size.
    """
    inputs_shape = get_tensor_shape(inputs, as_tensor)
    if data_format == 'channels_first':
        return inputs_shape[2:4]
    elif data_format == 'channels_last':
        return inputs_shape[1:3]
    else:
        raise Exception('unsupported data format')


def get_image_size_3d(inputs, data_format, as_tensor=False):
    """
    Return the 3D image_size for the given inputs and data_format.
    :param inputs: The inputs.
    :param data_format: 'channels_first' or 'channels_last'
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: The image_size.
    """
    inputs_shape = get_tensor_shape(inputs, as_tensor)
    if data_format == 'channels_first':
        return inputs_shape[2:5]
    elif data_format == 'channels_last':
        return inputs_shape[1:4]
    else:
        raise Exception('unsupported data format')


def get_channel_size(inputs, data_format, as_tensor=False):
    """
    Return the channel_size for the given inputs and data_format.
    :param inputs: The inputs.
    :param data_format: 'channels_first' or 'channels_last'
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: The channel_size.
    """
    inputs_shape = get_tensor_shape(inputs, as_tensor)
    if data_format == 'channels_first':
        return inputs_shape[1]
    elif data_format == 'channels_last':
        return inputs_shape[-1]


def get_batch_size(inputs, as_tensor=False):
    """
    Return the batch_size for the given inputs.
    :param inputs: The inputs.
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: The batch_size.
    """
    inputs_shape = get_tensor_shape(inputs, as_tensor)
    return inputs_shape[0]


def get_batch_channel_image_size(inputs, data_format, as_tensor=False):
    """
    Return the batch_size, channel_size and image_size for the given inputs and data_format.
    :param inputs: The inputs.
    :param data_format: 'channels_first' or 'channels_last'
    :param as_tensor: If True, return as a tensor, else return as a tuple.
    :return: Tuple of (batch_size, channel_size, image_size)
    """
    inputs_shape = get_tensor_shape(inputs, as_tensor)
    return get_batch_channel_image_size_from_shape_tuple(inputs_shape, data_format)


def get_batch_channel_image_size_from_shape_tuple(inputs_shape, data_format):
    """
    Return the batch_size, channel_size and image_size for the given inputs_shape and data_format.
    :param inputs_shape: The inputs shape.
    :param data_format: 'channels_first' or 'channels_last'
    :return: Tuple of (batch_size, channel_size, image_size)
    """
    tensor_dim = inputs_shape.shape[0] if isinstance(inputs_shape, tf.Tensor) else len(inputs_shape)
    if data_format == 'channels_first':
        if tensor_dim == 4:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:4]
        if tensor_dim == 5:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:5]
    elif data_format == 'channels_last':
        if tensor_dim == 4:
            return inputs_shape[0], inputs_shape[3], inputs_shape[1:3]
        if tensor_dim == 5:
            return inputs_shape[0], inputs_shape[4], inputs_shape[1:4]


def get_image_dimension(inputs):
    """
    Return the image dimension. 2 for 4D tensors, 3 for 5D tensors.
    :param inputs: The tensor.
    :return: 2 for 4D tensors, 3 for 5D tensors.
    """
    return inputs.shape.ndims - 2


def create_tensor_shape_tuple(batch_size, channel_size, image_size, data_format):
    """
    Create a shape tuple for the given values.
    :param batch_size: The batch size.
    :param channel_size: The channel size.
    :param image_size: The image size.
    :param data_format: 'channels_first' or 'channels_last'
    :return: Tuple of the shape.
    """
    if data_format == 'channels_first':
        return (batch_size, channel_size) + tuple(image_size)
    elif data_format == 'channels_last':
        return (batch_size,) + tuple(image_size) + (channel_size,)
    else:
        raise Exception('unsupported data format')


def channels_last_to_channels_first(inputs):
    """
    Transpose inputs from 'channels_last' to 'channels_first'.
    :param inputs: The tensor.
    :return: The transposed tensor.
    """
    if inputs.shape.ndims == 4:
        return tf.transpose(inputs, [0, 3, 1, 2])
    if inputs.shape.ndims == 5:
        return tf.transpose(inputs, [0, 4, 1, 2, 3])


def channels_first_to_channels_last(inputs):
    """
    Transpose inputs from 'channels_first' to 'channels_last'.
    :param inputs: The tensor.
    :return: The transposed tensor.
    """
    if inputs.shape.ndims == 4:
        return tf.transpose(inputs, [0, 2, 3, 1])
    if inputs.shape.ndims == 5:
        return tf.transpose(inputs, [0, 2, 3, 4, 1])
