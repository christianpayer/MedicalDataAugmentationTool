
import tensorflow as tf


def shift_with_padding(x, shift, axis, padding='zero'):
    """
    Shift axis and pad.
    :param x: Input tensor.
    :param shift: The amount of shift.
    :param axis: The axis to shift
    :param padding: The padding. 'zero', 'repeat', or 'mirror'.
    :return: The shifted and padded tensor.
    """
    if shift == 0:
        return x

    # compute tensor used for padding
    dim = x.shape.ndims
    abs_shift = abs(shift)
    if padding == 'zero':
        ones = tf.constant([1 if i != axis else 0 for i in range(dim)], tf.int32)
        shape = tf.shape(x) * ones + (1 - ones) * abs_shift
        x_padding = tf.zeros(shape, dtype=x.dtype)
    elif padding == 'repeat':
        if shift > 0:
            slices_pad = [slice(None) if i != axis else slice(None, 1) for i in range(dim)]
        else:
            slices_pad = [slice(None) if i != axis else slice(-1, None) for i in range(dim)]
        x_padding = x[slices_pad]
        x_padding = tf.concat([x_padding for _ in range(abs_shift)], axis=axis)
    elif padding == 'mirror':
        if shift > 0:
            slices_pad = [slice(None) if i != axis else slice(abs_shift - 1, None, -1) for i in range(dim)]
        else:
            slices_pad = [slice(None) if i != axis else slice(None, -abs_shift - 1, -1) for i in range(dim)]
        x_padding = x[slices_pad]
    else:
        raise ValueError('Invalid padding: ' + str(padding))

    # shift input tensor and pad with padding tensor
    if shift > 0:
        slices = [slice(None) if i != axis else slice(None, -abs_shift) for i in range(dim)]
        x_sliced = x[slices]
        return tf.concat([x_padding, x_sliced], axis=axis)
    else:
        slices = [slice(None) if i != axis else slice(abs_shift, None) for i in range(dim)]
        x_sliced = x[slices]
        return tf.concat([x_sliced, x_padding], axis=axis)


def generate_bb_mask(image_size, start, extent):
    """
    Generate a boolean mask with given image size and bounding box.
    :param image_size: The output image size.
    :param start: The start coordinate of the bounding box.
    :param extent: The extent of the bounding box.
    :return: Boolean tensor with
    """
    ranges = [tf.range(0, s) for s in image_size]
    coords = tf.meshgrid(*ranges, indexing='ij')
    mask = tf.ones(image_size, tf.bool)
    for i, c in enumerate(coords):
        mask = tf.math.logical_and(mask, c >= start[i])
        mask = tf.math.logical_and(mask, c < (start[i] + extent[i]))
    return mask


def reduce_mean_weighted(x, weights, axis=None, keepdims=False):
    """
    Calculate a weighted mean.
    :param x: The input tensor.
    :param weights: The weights. Must be broadcastable to x.
    :param axis: The axis to reduce.
    :param keepdims: If True, keep tensor dimensions.
    :return: The weighted mean.
    """
    input_weighted = x * weights
    sum_input_weighted = tf.reduce_sum(input_weighted, axis=axis, keepdims=keepdims)
    sum_weights = tf.reduce_sum(weights, axis=axis, keepdims=keepdims)
    return tf.math.divide_no_nan(sum_input_weighted, sum_weights)


def reduce_mean_loss_per_pixel(loss_per_pixel, weights=None, data_format='channels_first'):
    """
    Reduce mean loss per pixel to overal mean per batch.
    :param loss_per_pixel: Mean loss per pixel tensor. Must not have a channel dimension.
    :param weights: The optional weights per pixel. Must have a channel dimension.
    :param data_format: The data format.
    :return: Scalar tensor value.
    """
    image_axes = range(1, loss_per_pixel.shape.ndims)
    if weights is not None:
        channel_index = 1 if data_format == 'channels_first' else -1
        weights = tf.squeeze(weights, axis=channel_index)
        loss_per_channel = reduce_mean_weighted(loss_per_pixel, tf.cast(weights, tf.float32), axis=image_axes, keepdims=False)
    else:
        loss_per_channel = tf.reduce_mean(loss_per_pixel, axis=image_axes, keepdims=False)
    loss = tf.reduce_mean(loss_per_channel)
    return loss