
import tensorflow as tf

def get_tf_data_format(inputs, data_format):
    if inputs.shape.ndims == 4:
        return get_tf_data_format_2d(data_format)
    elif inputs.shape.ndims == 5:
        return get_tf_data_format_3d(data_format)
    else:
        raise Exception('unsupported inputs shape')


def get_tf_data_format_2d(data_format):
    if data_format == 'channels_first':
        return 'NCHW'
    elif data_format == 'channels_last':
        return 'NHWC'
    else:
        raise Exception('unsupported data format')


def get_tf_data_format_3d(data_format):
    if data_format == 'channels_first':
        return 'NCDHW'
    elif data_format == 'channels_last':
        return 'NDHWC'
    else:
        raise Exception('unsupported data format')


def get_channel_index(inputs, data_format):
    if inputs.shape.ndims == 4:
        return get_channel_index_2d(data_format)
    elif inputs.shape.ndims == 5:
        return get_channel_index_3d(data_format)
    else:
        raise Exception('unsupported inputs shape')


def get_channel_index_2d(data_format):
    if data_format == 'channels_first':
        return 1
    elif data_format == 'channels_last':
        return 3
    else:
        raise Exception('unsupported data format')


def get_channel_index_3d(data_format):
    if data_format == 'channels_first':
        return 1
    elif data_format == 'channels_last':
        return 4
    else:
        raise Exception('unsupported data format')


def get_image_axes(inputs, data_format):
    if inputs.shape.ndims == 4:
        if data_format == 'channels_first':
            return [2, 3]
        elif data_format == 'channels_last':
            return [1, 2]
    elif inputs.shape.ndims == 5:
        if data_format == 'channels_first':
            return [2, 3, 4]
        elif data_format == 'channels_last':
            return [1, 2, 3]


def get_image_size(inputs, data_format):
    if inputs.shape.ndims == 4:
        return get_image_size_2d(inputs, data_format)
    elif inputs.shape.ndims == 5:
        return get_image_size_3d(inputs, data_format)
    else:
        raise Exception('unsupported inputs shape')


def get_image_size_2d(inputs, data_format):
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        return inputs_shape[2:4]
    elif data_format == 'channels_last':
        return inputs_shape[1:3]
    else:
        raise Exception('unsupported data format')


def get_image_size_3d(inputs, data_format):
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        return inputs_shape[2:5]
    elif data_format == 'channels_last':
        return inputs_shape[1:4]
    else:
        raise Exception('unsupported data format')


def get_channel_size(inputs, data_format):
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        return inputs_shape[1]
    elif data_format == 'channels_last':
        return inputs_shape[-1]


def get_image_dimension(inputs):
    return inputs.shape.ndims - 2


def get_tensor_shape(batch_size, channel_size, image_size, data_format):
    if data_format == 'channels_first':
        return [batch_size, channel_size] + image_size
    elif data_format == 'channels_last':
        return [batch_size] + image_size + [channel_size]
    else:
        raise Exception('unsupported data format')


def get_batch_channel_image_size(inputs, data_format):
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        if len(inputs_shape) == 4:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:4]
        if len(inputs_shape) == 5:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:5]
    elif data_format == 'channels_last':
        if len(inputs_shape) == 4:
            return inputs_shape[0], inputs_shape[3], inputs_shape[1:3]
        if len(inputs_shape) == 5:
            return inputs_shape[0], inputs_shape[4], inputs_shape[1:4]


def channels_last_to_channels_first(inputs):
    if inputs.shape.ndims == 4:
        return tf.transpose(inputs, [0, 3, 1, 2])
    if inputs.shape.ndims == 5:
        return tf.transpose(inputs, [0, 4, 1, 2, 3])


def channels_first_to_channels_last(inputs):
    if inputs.shape.ndims == 4:
        return tf.transpose(inputs, [0, 2, 3, 1])
    if inputs.shape.ndims == 5:
        return tf.transpose(inputs, [0, 2, 3, 4, 1])