
import tensorflow as tf
import numpy as np


def generate_heatmap_target(heatmap_size, landmarks, sigmas, scale=1.0, normalize=False, data_format='channels_first'):
    """
    Generates heatmap images for the given parameters.
    :param heatmap_size: The image size of a single heatmap.
    :param landmarks: The list of landmarks. For each landmark, a heatmap on the given coordinate will be generated. If landmark.is_valid is False, then the heatmap will be empty.
    :param sigmas: The sigmas for the individual heatmaps. May be either fixed, or trainable.
    :param scale: The scale factor for each heatmap. Each pixel value will be multiplied by this value.
    :param normalize: If true, each heatmap value will be multiplied by the normalization factor of the gaussian.
    :param data_format: The data format of the resulting tensor of heatmap images.
    :return: The tensor of heatmap images.
    """
    landmarks_shape = landmarks.get_shape().as_list()
    sigmas_shape = sigmas.get_shape().as_list()
    batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[1]
    dim = landmarks_shape[2] - 1
    assert len(heatmap_size) == dim, 'Dimensions do not match.'
    assert sigmas_shape[0] == num_landmarks, 'Number of sigmas does not match.'

    if data_format == 'channels_first':
        heatmap_axis = 1
        landmarks_reshaped = tf.reshape(landmarks[..., 1:], [batch_size, num_landmarks] + [1] * dim + [dim])
        is_valid_reshaped = tf.reshape(landmarks[..., 0], [batch_size, num_landmarks] + [1] * dim)
        sigmas_reshaped = tf.reshape(sigmas, [1, num_landmarks] + [1] * dim)
    else:
        heatmap_axis = dim + 1
        landmarks_reshaped = tf.reshape(landmarks[..., 1:], [batch_size] + [1] * dim + [num_landmarks, dim])
        is_valid_reshaped = tf.reshape(landmarks[..., 0], [batch_size] + [1] * dim + [num_landmarks])
        sigmas_reshaped = tf.reshape(sigmas, [1] + [1] * dim + [num_landmarks])

    aranges = [np.arange(s) for s in heatmap_size]
    grid = tf.meshgrid(*aranges, indexing='ij')

    grid_stacked = tf.stack(grid, axis=dim)
    grid_stacked = tf.cast(grid_stacked, tf.float32)
    grid_stacked = tf.stack([grid_stacked] * batch_size, axis=0)
    grid_stacked = tf.stack([grid_stacked] * num_landmarks, axis=heatmap_axis)

    if normalize:
        scale /= tf.pow(np.sqrt(2 * np.pi) * sigmas_reshaped, dim)

    squared_distances = tf.reduce_sum(tf.pow(grid_stacked - landmarks_reshaped, 2.0), axis=-1)
    heatmap = scale * tf.exp(-squared_distances / (2 * tf.pow(sigmas_reshaped, 2)))
    heatmap_or_zeros = tf.where((is_valid_reshaped + tf.zeros_like(heatmap)) > 0, heatmap, tf.zeros_like(heatmap))

    return heatmap_or_zeros
