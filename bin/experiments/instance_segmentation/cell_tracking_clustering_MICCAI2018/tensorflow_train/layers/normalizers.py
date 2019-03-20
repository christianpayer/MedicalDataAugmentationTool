
import tensorflow as tf
from tensorflow_train.utils.data_format import get_channel_index, get_image_axes, get_tf_data_format_2d


def instance_norm(inputs, is_training, name='', data_format='channels_first', epsilon=1e-5, beta_initializer=tf.constant_initializer(0.0), gamma_initializer=tf.constant_initializer(1.0)):
    with tf.variable_scope(name):
        channel_index = get_channel_index(inputs, data_format)
        image_axes = get_image_axes(inputs, data_format=data_format)
        depth = inputs.get_shape()[channel_index]
        mean, variance = tf.nn.moments(inputs, axes=image_axes, keep_dims=True)
        inv = tf.rsqrt(variance + epsilon)
        normalized = (inputs - mean) * inv
        offset = tf.get_variable('offset', [depth], trainable=is_training, initializer=beta_initializer)
        scale = tf.get_variable('scale', [depth], trainable=is_training, initializer=gamma_initializer)
        offset_scale_shape = [1] * inputs.shape.ndims
        offset_scale_shape[channel_index] = depth
        offset = tf.reshape(offset, offset_scale_shape)
        scale = tf.reshape(scale, offset_scale_shape)
        return scale * normalized + offset


def batch_norm(inputs, is_training, name='', data_format='channels_first'):
    # use faster fused batch_norm for 4 channel tensors
    if len(inputs.shape.ndims) == 4:
        data_format_tf = get_tf_data_format_2d(data_format)
        return tf.contrib.layers.batch_norm(inputs, is_training=is_training, data_format=data_format_tf, fused=True, scope=name + '/bn')
    elif len(inputs.shape.ndims) == 5:
        return tf.layers.batch_normalization(inputs, axis=1, name=name + '/bn', training=is_training)
    else:
        raise Exception('This batch_norm only supports images. Use batch_norm_dense or basic tensorflow version instead.')


# def layer_norm(inputs, is_training, name='', data_format='channels_first'):
#     assert data_format == 'channels_last', 'layer_norm only supports data_format == \'channels_last\''
#     return tf.contrib.layers.layer_norm(inputs, scope=name)


def layer_norm(inputs, is_training, name='', data_format='channels_first'):
    with tf.variable_scope(name):
        inputs_shape = inputs.get_shape().as_list()
        channel_index = get_channel_index(inputs, data_format)
        params_shape = [1] * len(inputs_shape)
        params_shape[channel_index] = inputs_shape[channel_index]
        # Allocate parameters for the beta and gamma of the normalization.
        beta = tf.get_variable('beta', shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=is_training)
        gamma = tf.get_variable('gamma', shape=params_shape, dtype=tf.float32, initializer=tf.ones_initializer(), trainable=is_training)
        norm_axes = list(range(1, len(inputs_shape)))
        mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        outputs = tf.nn.batch_normalization(inputs, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-12)
        return outputs


def batch_norm_dense(inputs, is_training, name=''):
    return tf.layers.batch_normalization(inputs, axis=1, name=name + '/bn', training=is_training)