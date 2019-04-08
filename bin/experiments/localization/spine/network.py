
import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, avg_pool3d, dropout, add
from tensorflow_train.layers.interpolation import upsample3d_linear, upsample3d_cubic
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.networks.unet import UnetClassic3D


class SCNetLocal(UnetBase):
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


def network_scn(input, num_heatmaps, is_training, data_format='channels_first'):
    num_filters_base = 64
    activation = lambda x, name: tf.nn.leaky_relu(x, name=name, alpha=0.1)
    padding = 'reflect'
    heatmap_layer_kernel_initializer = tf.truncated_normal_initializer(stddev=0.001)
    downsampling_factor = 8
    node = conv3d(input,
                  filters=num_filters_base,
                  kernel_size=[3, 3, 3],
                  name='conv0',
                  activation=activation,
                  data_format=data_format,
                  is_training=is_training)
    scnet_local = SCNetLocal(num_filters_base=num_filters_base,
                             num_levels=4,
                             double_filters_per_level=False,
                             normalization=None,
                             activation=activation,
                             data_format=data_format,
                                      padding=padding)
    unet_out = scnet_local(node, is_training)
    local_heatmaps = conv3d(unet_out,
                            filters=num_heatmaps,
                            kernel_size=[3, 3, 3],
                            name='local_heatmaps',
                            kernel_initializer=heatmap_layer_kernel_initializer,
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)
    downsampled = avg_pool3d(local_heatmaps, [downsampling_factor] * 3, name='local_downsampled', data_format=data_format)
    conv = conv3d(downsampled, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv0', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv1', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv2', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_heatmaps, kernel_size=[7, 7, 7], name='spatial_downsampled', kernel_initializer=heatmap_layer_kernel_initializer, activation=tf.nn.tanh, data_format=data_format, is_training=is_training, padding=padding)
    spatial_heatmaps = upsample3d_cubic(conv, [downsampling_factor] * 3, name='spatial_heatmaps', data_format=data_format, padding='valid_cropped')

    heatmaps = local_heatmaps * spatial_heatmaps

    return heatmaps, local_heatmaps, spatial_heatmaps


def network_unet(input, num_heatmaps, is_training, data_format='channels_first'):
    num_filters_base = 64
    activation = tf.nn.relu
    node = conv3d(input,
                  filters=num_filters_base,
                  kernel_size=[3, 3, 3],
                  name='conv0',
                  activation=activation,
                  data_format=data_format,
                  is_training=is_training)
    scnet_local = UnetClassic3D(num_filters_base=num_filters_base,
                             num_levels=5,
                             double_filters_per_level=False,
                             normalization=None,
                             activation=activation,
                             data_format=data_format)
    unet_out = scnet_local(node, is_training)
    heatmaps = conv3d(unet_out,
                            filters=num_heatmaps,
                            kernel_size=[3, 3, 3],
                            name='heatmaps',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)

    return heatmaps, heatmaps, heatmaps
