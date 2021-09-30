import tensorflow as tf
from tensorflow_train_v2.utils.data_format import get_image_axes, get_channel_index, get_batch_channel_image_size, channels_first_to_channels_last, channels_last_to_channels_first
from tensorflow_train_v2.utils.tensorflow_util import reduce_mean_weighted, reduce_mean_loss_per_pixel


def sigmoid_cross_entropy_with_logits(labels, logits, labels_mask=None, weights=None, data_format='channels_first'):
    if labels_mask is not None:
        labels_mask = tf.cast(labels_mask, dtype=labels.dtype)
        labels *= labels_mask
        logits *= labels_mask

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.cast(logits, tf.float32))
    if weights is not None:
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def generalized_dice_loss(labels, logits=None, logits_as_probability=None, labels_mask=None, data_format='channels_first', weights=None, weight_labels=True, squared=True, weight_epsilon=1e-04, epsilon=1e-08):
    """
    Taken from Generalised Dice overlap as a deep learning loss function for
    highly unbalanced segmentations (https://arxiv.org/pdf/1707.03237.pdf)
    :param labels: groundtruth labels
    :param logits: network predictions
    :param labels_mask: binary mask used to define regions in which the loss is computed (1 = compute loss, 0 = don't compute loss)
    :param data_format: either 'channels_first' of 'channels_last'
    :param epsilon: used for numerical instabilities caused by denominator = 0
    :return: Tensor of mean generalized dice loss of all images of the batch
    """
    assert (logits is None and logits_as_probability is not None) or (logits is not None and logits_as_probability is None), 'Set either logits or logits_as_probability, but not both.'
    dtype = logits.dtype if logits is not None else logits_as_probability.dtype
    labels = labels if labels.dtype == dtype else tf.cast(labels, dtype)
    channel_index = get_channel_index(labels, data_format)
    image_axes = get_image_axes(labels, data_format)
    labels_shape = labels.get_shape().as_list()
    num_labels = labels_shape[channel_index]
    # calculate logits propability as softmax (p_n)
    if logits_as_probability is None:
        logits_as_probability = tf.nn.softmax(logits, axis=channel_index)
    if weight_labels:
        # calculate label weights (w_l)
        label_weights = tf.constant(1, dtype=dtype) / (tf.reduce_sum(labels, axis=image_axes) ** 2 + tf.constant(weight_epsilon, dtype=dtype))
    else:
        label_weights = tf.constant(1, dtype=dtype)

    if labels_mask is not None:
        labels_mask = tf.cast(labels_mask, dtype=labels.dtype)
        labels *= labels_mask
        logits_as_probability *= labels_mask

    # GDL_b based on equation in reference paper
    numerator = tf.reduce_sum(label_weights * tf.reduce_sum(labels * logits_as_probability, axis=image_axes), axis=1)
    if squared:
        # square logits, no need to square labels, as they are either 0 or 1
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + (logits_as_probability**2), axis=image_axes), axis=1)
    else:
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + logits_as_probability, axis=image_axes), axis=1)
    loss = 1 - 2 * (numerator + epsilon) / (denominator + epsilon)

    if weights is not None:
        channel_index = get_channel_index(weights, data_format)
        weights = tf.squeeze(weights, axis=channel_index)
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)
