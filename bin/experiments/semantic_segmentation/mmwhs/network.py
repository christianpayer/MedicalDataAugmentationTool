
import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, concat_channels, avg_pool3d
from tensorflow_train.layers.interpolation import upsample3d_linear
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.layers.initializers import he_initializer


class UnetClassicAvgLinear3d(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def downsample(self, node, current_level, is_training):
        return avg_pool3d(node, [2, 2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample3d_linear(node, [2, 2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv3d(node,
                      self.num_filters(current_level),
                      [3, 3, 3],
                      name='conv' + postfix,
                      activation=self.activation,
                      normalization=None,
                      is_training=is_training,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      padding=self.padding)


def network_scn(input, num_labels, is_training, data_format='channels_first'):
    downsampling_factor = 4
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = he_initializer
    spatial_activation = None
    padding = 'reflect'
    with tf.variable_scope('unet'):
        unet = UnetClassicAvgLinear3d(64, 4, data_format=data_format, double_filters_per_level=True, kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        local_prediction = unet(input, is_training=is_training)
        local_prediction = conv3d(local_prediction, num_labels, [1, 1, 1], name='local_prediction', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
    with tf.variable_scope('spatial_configuration'):
        local_prediction_pool = avg_pool3d(local_prediction, [downsampling_factor] * 3, name='local_prediction_pool')
        scconv = conv3d(local_prediction_pool, 64, [5, 5, 5], name='scconv0', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [5, 5, 5], name='scconv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [5, 5, 5], name='scconv2', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        spatial_prediction_pool = conv3d(scconv, num_labels, [5, 5, 5], name='spatial_prediction_pool', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)
        spatial_prediction = upsample3d_linear(spatial_prediction_pool, [downsampling_factor] * 3, name='spatial_prediction', padding='valid_cropped')
    with tf.variable_scope('combination'):
        prediction = local_prediction * spatial_prediction
    return prediction, local_prediction, spatial_prediction


def network_unet(input, num_labels, is_training, data_format='channels_first'):
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = None
    padding = 'reflect'
    with tf.variable_scope('unet'):
        unet = UnetClassicAvgLinear3d(64, 4, data_format=data_format, double_filters_per_level=True, kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        prediction = unet(input, is_training=is_training)
        prediction = conv3d(prediction, num_labels, [1, 1, 1], name='output', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
    return prediction, prediction, prediction
