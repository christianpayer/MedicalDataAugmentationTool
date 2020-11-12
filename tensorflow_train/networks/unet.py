
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import concat_channels, add, dropout
from tensorflow_train.networks.unet_base import UnetBase, UnetBase2D, UnetBase3D


class UnetClassic(UnetBase):
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


class UnetAdd(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

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


class UnetParallel(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        return self.conv(node, current_level, '', is_training)

    def parallel_block(self, node, current_level, is_training):
        return self.conv(node, current_level, '', is_training)

    def expanding_block(self, node, current_level, is_training):
        return self.conv(node, current_level, '', is_training)


class UnetParallelDropout(UnetBase):
    def __init__(self,
                 num_filters_base,
                 num_levels,
                 dropout_rate,
                 double_filters_per_level=False,
                 normalization=None,
                 activation=tf.nn.relu):
        super(UnetParallelDropout, self).__init__(num_filters_base,
                                                  num_levels,
                                                  double_filters_per_level,
                                                  normalization,
                                                  activation)
        self.dropout_rate = dropout_rate

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return dropout(node, self.dropout_rate, name='dropout', is_training=is_training)

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return dropout(node, self.dropout_rate, name='dropout', is_training=is_training)

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return dropout(node, self.dropout_rate, name='dropout', is_training=is_training)


class UnetClassic2D(UnetClassic, UnetBase2D): pass
class UnetClassic3D(UnetClassic, UnetBase3D): pass
class UnetAdd2D(UnetAdd, UnetBase2D): pass
class UnetAdd3D(UnetAdd, UnetBase3D): pass
class UnetParallel2D(UnetParallel, UnetBase2D): pass
class UnetParallel3D(UnetParallel, UnetBase3D): pass
class UnetParallelDropout2D(UnetParallelDropout, UnetBase2D): pass
class UnetParallelDropout3D(UnetParallelDropout, UnetBase3D): pass