
import tensorflow as tf
from tensorflow_train.utils.data_format import get_image_axes, get_channel_index
from tensorflow_train.utils.tensorflow_util import reduce_mean_weighted


def softmax_cross_entropy_with_logits(labels, logits, weights=None, data_format='channels_first'):
    channel_index = get_channel_index(labels, data_format)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=channel_index)
    if weights is not None:
        channel_index = get_channel_index(weights, data_format)
        weights = tf.squeeze(weights, axis=channel_index)
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def generalized_dice_loss(labels, logits=None, logits_as_probability=None, data_format='channels_first', weights=None, weight_labels=True, squared=True, weight_epsilon=1e-08, epsilon=1e-08):
    """
    Taken from Generalised Dice overlap as a deep learning loss function for
    highly unbalanced segmentations (https://arxiv.org/pdf/1707.03237.pdf)
    :param labels: groundtruth labels
    :param logits: network predictions
    :param data_format: either 'channels_first' of 'channels_last'
    :param epsilon: used for numerical instabilities caused by denominator = 0
    :return: Tensor of mean generalized dice loss of all images of the batch
    """
    assert (logits is None and logits_as_probability is not None) or (logits is not None and logits_as_probability is None), 'Set either logits or logits_as_probability, but not both.'
    channel_index = get_channel_index(labels, data_format)
    image_axes = get_image_axes(labels, data_format)
    labels_shape = labels.get_shape().as_list()
    num_labels = labels_shape[channel_index]
    # calculate logits propability as softmax (p_n)
    if logits_as_probability is None:
        logits_as_probability = tf.nn.softmax(logits, dim=channel_index)
    if weight_labels:
        # calculate label weights (w_l)
        label_weights = 1 / (tf.reduce_sum(labels, axis=image_axes) ** 2 + weight_epsilon)
    else:
        label_weights = 1
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
