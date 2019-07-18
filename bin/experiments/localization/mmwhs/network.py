
import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, concat_channels, avg_pool3d
from tensorflow_train.layers.interpolation import upsample3d_linear
from tensorflow_train.layers.initializers import he_initializer
from tensorflow_train.networks.unet import UnetBase


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


def network_unet(input, num_landmarks, is_training, num_filters=64, num_levels=4, data_format='channels_first'):
    padding = 'same'
    kernel_initializer = he_initializer
    heatmap_initializer = tf.initializers.truncated_normal(stddev=0.0001)
    activation = tf.nn.relu
    heatmap_activation = None
    with tf.variable_scope('unet'):
        unet = UnetClassicAvgLinear3d(num_filters_base=num_filters, num_levels=num_levels, activation=activation, kernel_initializer=kernel_initializer, data_format=data_format, padding=padding, use_addition=False, use_avg_pooling=True, use_linear_upsampling=True)
        node = unet(input, is_training=is_training)
        heatmaps = conv3d(node, num_landmarks, kernel_size=[1, 1, 1], name='heatmaps', activation=heatmap_activation, kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
    return heatmaps
