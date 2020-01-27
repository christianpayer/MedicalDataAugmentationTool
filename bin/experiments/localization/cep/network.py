
import tensorflow as tf
from tensorflow_train.layers.layers import conv2d, avg_pool2d, concat_channels, dropout, add
from tensorflow_train.layers.interpolation import upsample2d_linear, upsample2d_cubic
from tensorflow_train.layers.initializers import he_initializer
from tensorflow_train.networks.unet import UnetClassic2D
from tensorflow_train.networks.unet_base import UnetBase


def network_unet(input, num_landmarks, is_training, data_format='channels_first'):
    num_filters = 128
    num_levels = 5
    padding = 'same'
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    heatmap_activation = None
    heatmap_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    with tf.variable_scope('unet'):
        unet = UnetClassic2D(num_filters, num_levels, activation=activation, kernel_initializer=kernel_initializer, data_format=data_format, padding=padding)
        node = unet(input, is_training=is_training)
        heatmaps = conv2d(node, num_landmarks, kernel_size=[1, 1], name='heatmaps', activation=heatmap_activation, kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
    return heatmaps


class UnetClassicAvgLinear2D(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = dropout(node, 0.5, 'drop' + str(current_level), is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = dropout(node, 0.5, 'drop' + str(current_level), is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def downsample(self, node, current_level, is_training):
        return avg_pool2d(node, [2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample2d_linear(node, factors=[2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv2d(node,
                      self.num_filters(current_level),
                      [3, 3],
                      name='conv' + postfix,
                      activation=self.activation,
                      normalization=None,
                      is_training=is_training,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      padding=self.padding)


class SCNetLocal(UnetBase):
    def downsample(self, node, current_level, is_training):
        return avg_pool2d(node, [2] * 2, name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample2d_linear(node, factors=[2] * 2, name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv2d(node,
                      self.num_filters(current_level),
                      [3] * 2,
                      name='conv' + postfix,
                      activation=self.activation,
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format,
                      padding=self.padding)

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '_0', is_training)
        node = dropout(node, 0.5, 'drop' + str(current_level), is_training)
        node = self.conv(node, current_level, '_1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        return node


def network_scn_mia(input, num_landmarks, is_training, data_format='channels_first'):
    num_filters_base = 128
    activation = lambda x, name: tf.nn.leaky_relu(x, name=name, alpha=0.1)
    padding = 'same'
    heatmap_layer_kernel_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    downsampling_factor = 16
    dim = 2
    node = conv2d(input,
                  filters=num_filters_base,
                  kernel_size=[3] * dim,
                  name='conv0',
                  activation=activation,
                  kernel_initializer=he_initializer,
                  data_format=data_format,
                  is_training=is_training)
    scnet_local = SCNetLocal(num_filters_base=num_filters_base,
                             num_levels=4,
                             double_filters_per_level=False,
                             normalization=None,
                             kernel_initializer=he_initializer,
                             activation=activation,
                             data_format=data_format,
                             padding=padding)
    unet_out = scnet_local(node, is_training)
    local_heatmaps = conv2d(unet_out,
                            filters=num_landmarks,
                            kernel_size=[3] * dim,
                            name='local_heatmaps',
                            kernel_initializer=heatmap_layer_kernel_initializer,
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)
    downsampled = avg_pool2d(local_heatmaps, [downsampling_factor] * dim, name='local_downsampled', data_format=data_format)
    conv = conv2d(downsampled, filters=num_filters_base, kernel_size=[11] * dim, kernel_initializer=he_initializer, name='sconv0', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv2d(conv, filters=num_filters_base, kernel_size=[11] * dim, kernel_initializer=he_initializer, name='sconv1', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv2d(conv, filters=num_filters_base, kernel_size=[11] * dim, kernel_initializer=he_initializer, name='sconv2', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv2d(conv, filters=num_landmarks, kernel_size=[11] * dim, name='spatial_downsampled', kernel_initializer=heatmap_layer_kernel_initializer, activation=tf.nn.tanh, data_format=data_format, is_training=is_training, padding=padding)
    spatial_heatmaps = upsample2d_cubic(conv, factors=[downsampling_factor] * dim, name='spatial_heatmaps', data_format=data_format, padding='valid_cropped')

    heatmaps = local_heatmaps * spatial_heatmaps

    return heatmaps
