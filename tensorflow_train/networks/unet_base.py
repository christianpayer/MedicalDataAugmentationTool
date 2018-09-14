
import tensorflow as tf
from tensorflow_train.layers.layers import conv2d, max_pool2d, upsample2d, conv3d, max_pool3d, upsample3d
from tensorflow_train.layers.initializers import he_initializer


class UnetBase(object):
    def __init__(self,
                 num_filters_base,
                 num_levels,
                 double_filters_per_level=False,
                 normalization=None,
                 activation=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 data_format='channels_first',
                 padding='same',
                 **kwargs):
        self.num_filters_base = num_filters_base
        self.num_levels = num_levels
        self.double_filters_per_level = double_filters_per_level
        self.normalization = normalization
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.data_format = data_format
        self.padding = padding

    def num_filters(self, current_level):
        if self.double_filters_per_level:
            return self.num_filters_base * (2 ** current_level)
        else:
            return self.num_filters_base

    def downsample(self, node, current_level, is_training):
        raise NotImplementedError

    def upsample(self, node, current_level, is_training):
        raise NotImplementedError

    def conv(self, node, current_level, postfix, is_training):
        raise NotImplementedError

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        raise NotImplementedError

    def contracting_block(self, node, current_level, is_training):
        raise NotImplementedError

    def parallel_block(self, node, current_level, is_training):
        raise NotImplementedError

    def expanding_block(self, node, current_level, is_training):
        raise NotImplementedError

    def contracting(self, node, is_training):
        with tf.variable_scope('contracting'):
            print('contracting path')
            contracting_level_nodes = []
            for current_level in range(self.num_levels):
                with tf.variable_scope('level' + str(current_level)):
                    node = self.contracting_block(node, current_level, is_training)
                contracting_level_nodes.append(node)
                # perform downsampling, if not at last level
                if current_level < self.num_levels - 1:
                    node = self.downsample(node, current_level, is_training)
            return contracting_level_nodes

    def parallel(self, contracting_level_nodes, is_training):
        with tf.variable_scope('parallel'):
            print('parallel path')
            parallel_level_nodes = []
            for current_level in range(self.num_levels):
                with tf.variable_scope('level' + str(current_level)):
                    node = self.parallel_block(contracting_level_nodes[current_level], current_level, is_training)
                parallel_level_nodes.append(node)
            return parallel_level_nodes

    def expanding(self, parallel_level_nodes, is_training):
        with tf.variable_scope('expanding'):
            print('expanding path')
            node = None
            for current_level in reversed(range(self.num_levels)):
                if current_level == self.num_levels - 1:
                    # on deepest level, do not combine nodes
                    node = parallel_level_nodes[current_level]
                else:
                    node = self.upsample(node, current_level, is_training)
                    node = self.combine(parallel_level_nodes[current_level], node, current_level, is_training)
                with tf.variable_scope('level' + str(current_level)):
                    node = self.expanding_block(node, current_level, is_training)
            return node

    def __call__(self, node, is_training):
        return self.expanding(self.parallel(self.contracting(node, is_training), is_training), is_training)


class UnetBase3D(UnetBase):
    def downsample(self, node, current_level, is_training):
        return max_pool3d(node, [2, 2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample3d(node, [2, 2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv3d(node,
                      self.num_filters(current_level),
                      [3, 3, 3],
                      name='conv' + postfix,
                      activation=self.activation,
                      kernel_initializer=self.kernel_initializer,
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format,
                      padding=self.padding)


class UnetBase2D(UnetBase):
    def downsample(self, node, current_level, is_training):
        return max_pool2d(node, [2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample2d(node, [2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv2d(node,
                      self.num_filters(current_level),
                      [3, 3],
                      name='conv' + postfix,
                      activation=self.activation,
                      kernel_initializer=self.kernel_initializer,
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format,
                      padding=self.padding)
