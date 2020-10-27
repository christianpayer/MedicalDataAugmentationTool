
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
