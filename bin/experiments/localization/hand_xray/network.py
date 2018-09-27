
import tensorflow as tf
from tensorflow_train.utils.data_format import get_channel_index
from tensorflow_train.layers.layers import conv2d, avg_pool2d, concat_channels
from tensorflow_train.layers.interpolation import upsample2d_linear
from tensorflow_train.layers.initializers import he_initializer
from tensorflow_train.networks.unet import UnetClassic2D
from tensorflow_train.networks.unet_base import UnetBase


def network_scn(input, num_landmarks, is_training, data_format='channels_first'):
    num_filters = 128
    local_kernel_size = [5, 5]
    spatial_kernel_size = [15, 15]
    downsampling_factor = 8
    padding = 'same'
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    heatmap_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    local_activation = None
    spatial_activation = None
    with tf.variable_scope('local_appearance'):
        node = conv2d(input, num_filters, kernel_size=local_kernel_size, name='conv1', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=local_kernel_size, name='conv2', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=local_kernel_size, name='conv3', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        local_heatmaps = conv2d(node, num_landmarks, kernel_size=local_kernel_size, name='local_heatmaps', activation=local_activation,  kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
    with tf.variable_scope('spatial_configuration'):
        local_heatmaps_downsampled = avg_pool2d(local_heatmaps, [downsampling_factor, downsampling_factor], name='local_heatmaps_downsampled', data_format=data_format)
        channel_axis = get_channel_index(local_heatmaps_downsampled, data_format)
        local_heatmaps_downsampled_split = tf.split(local_heatmaps_downsampled, num_landmarks, channel_axis)
        spatial_heatmaps_downsampled_split = []
        for i in range(num_landmarks):
            local_heatmaps_except_i = tf.concat([local_heatmaps_downsampled_split[j] for j in range(num_landmarks) if i != j], name='h_app_except_'+str(i), axis=channel_axis)
            h_acc = conv2d(local_heatmaps_except_i, 1, kernel_size=spatial_kernel_size, name='h_acc_'+str(i), activation=spatial_activation, kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
            spatial_heatmaps_downsampled_split.append(h_acc)
        spatial_heatmaps_downsampled = tf.concat(spatial_heatmaps_downsampled_split, name='spatial_heatmaps_downsampled', axis=channel_axis)
        spatial_heatmaps = upsample2d_linear(spatial_heatmaps_downsampled, [downsampling_factor, downsampling_factor], name='spatial_prediction', padding='valid_cropped', data_format=data_format)
    with tf.variable_scope('combination'):
        heatmaps = local_heatmaps * spatial_heatmaps
    return heatmaps


def network_downsampling(input, num_landmarks, is_training, data_format='channels_first'):
    num_filters = 128
    kernel_size = [5, 5]
    num_levels = 3
    padding = 'same'
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    heatmap_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    heatmap_activation = None
    node = input
    with tf.variable_scope('downsampling'):
        for i in range(num_levels):
            with tf.variable_scope('level' + str(i)):
                node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv0', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
                node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv1', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
                if i != num_levels - 1:
                    node = avg_pool2d(node, [2, 2], name='downsampling', data_format=data_format)
        heatmaps = conv2d(node, num_landmarks, kernel_size=[1, 1], name='heatmaps', activation=heatmap_activation, kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
    return heatmaps


def network_conv(input, num_landmarks, is_training, data_format='channels_first'):
    num_filters = 128
    kernel_size = [11, 11]
    padding = 'same'
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    heatmap_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    heatmap_activation = None
    node = input
    with tf.variable_scope('downsampling'):
        node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv0', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv1', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv2', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv3', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv4', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        node = conv2d(node, num_filters, kernel_size=kernel_size, name='conv5', activation=activation, kernel_initializer=kernel_initializer, padding=padding, data_format=data_format, is_training=is_training)
        heatmaps = conv2d(node, num_landmarks, kernel_size=[1, 1], name='heatmaps', activation=heatmap_activation, kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
    return heatmaps


def network_unet(input, num_landmarks, is_training, data_format='channels_first'):
    num_filters = 128
    num_levels = 4
    padding = 'same'
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    heatmap_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    heatmap_activation = None
    with tf.variable_scope('unet'):
        unet = UnetClassic2D(num_filters, num_levels, activation=activation, kernel_initializer=kernel_initializer, data_format=data_format, padding=padding)
        node = unet(input, is_training=is_training)
        heatmaps = conv2d(node, num_landmarks, kernel_size=[1, 1], name='heatmaps', activation=heatmap_activation, kernel_initializer=heatmap_initializer, padding=padding, data_format=data_format, is_training=is_training)
    return heatmaps


### The following network is from our paper for the MMWHS challenge

class UnetClassicAvgLinear2D(UnetBase):
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
        return avg_pool2d(node, [2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample2d_linear(node, [2, 2], name='upsample' + str(current_level), data_format=self.data_format)

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


def network_scn_mmwhs(input, num_landmarks, is_training, data_format='channels_first'):
    downsampling_factor = 8
    num_filters = 128
    num_levels = 4
    spatial_kernel_size = [5, 5]
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = tf.truncated_normal_initializer(stddev=0.0001)
    spatial_activation = None
    padding = 'reflect'
    with tf.variable_scope('unet'):
        unet = UnetClassicAvgLinear2D(num_filters, num_levels, data_format=data_format, double_filters_per_level=False, kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        local_prediction = unet(input, is_training=is_training)
        local_prediction = conv2d(local_prediction, num_landmarks, [1, 1], name='local_prediction', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
    with tf.variable_scope('spatial_configuration'):
        local_prediction_pool = avg_pool2d(local_prediction, [downsampling_factor] * 2, name='local_prediction_pool')
        scconv = conv2d(local_prediction_pool, num_filters, spatial_kernel_size, name='scconv0', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv2d(scconv, num_filters, spatial_kernel_size, name='scconv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv2d(scconv, num_filters, spatial_kernel_size, name='scconv2', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        spatial_prediction_pool = conv2d(scconv, num_landmarks, spatial_kernel_size, name='spatial_prediction_pool', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)
        spatial_prediction = upsample2d_linear(spatial_prediction_pool, [downsampling_factor] * 2, name='spatial_prediction', padding='valid_cropped')
    with tf.variable_scope('combination'):
        prediction = local_prediction * spatial_prediction
    return prediction
