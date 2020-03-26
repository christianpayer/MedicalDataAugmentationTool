import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, add, max_pool3d, upsample3d, concat_channels, conv2d, max_pool2d, upsample2d
from tensorflow_train.layers.initializers import selu_initializer, he_initializer
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.networks.unet_base import UnetBase3D, UnetBase2D

class HourglassNet3D(UnetBase3D):
    def downsample(self, node, current_level, is_training):
        return max_pool3d(node, [1, 2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample3d(node, [1, 2, 2], name='upsample' + str(current_level), data_format=self.data_format)

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


    def combine(self, parallel_node, upsample_node, current_level, is_training):
        node = add([parallel_node, upsample_node], name='concat' + str(current_level))
        return node

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node


class HourglassNet2D(UnetBase2D):
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


    def combine(self, parallel_node, upsample_node, current_level, is_training):
        node = add([parallel_node, upsample_node], name='concat' + str(current_level))
        return node

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node


def network(input, is_training, num_outputs_embedding, actual_network, filters=64, levels=7, activation='relu', normalize=False, data_format='channels_first', padding='same'):
    if activation == 'selu':
        activation = tf.nn.selu
        kernel_initializer = selu_initializer
    elif activation == 'relu':
        activation = tf.nn.relu
        kernel_initializer = he_initializer
    elif activation == 'tanh':
        activation = tf.nn.tanh
        kernel_initializer = selu_initializer
    padding = padding
    embedding_axis = 1 if data_format == 'channels_first' else 4
    if normalize:
        embeddings_activation = lambda x, name: tf.nn.l2_normalize(x, dim=embedding_axis, name=name, epsilon=1e-4)
    else:
        if activation == tf.nn.selu:
            embeddings_activation = tf.nn.selu
        else:
            embeddings_activation = None

    embeddings_normalization = lambda x, name: tf.nn.l2_normalize(x, dim=embedding_axis, name=name, epsilon=1e-4)

    with tf.variable_scope('unet_0'):
        unet = actual_network(num_filters_base=filters, kernel=[3, 3, 3], num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet', padding=padding)
        unet_out = unet(input, is_training)
        embeddings = conv3d(unet_out, kernel_size=[1, 1, 1], name='embeddings', filters=num_outputs_embedding, kernel_initializer=kernel_initializer, activation=embeddings_activation, data_format=data_format, is_training=is_training, padding=padding)
    with tf.variable_scope('unet_1'):
        normalized_embeddings = embeddings_normalization(embeddings, 'embeddings_normalized')
        input_concat = concat_channels([input, normalized_embeddings], name='input_concat', data_format=data_format)
        unet = actual_network(num_filters_base=filters, kernel=[3, 3, 3], num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet', padding=padding)
        unet_out = unet(input_concat, is_training)
        embeddings_2 = conv3d(unet_out, kernel_size=[1, 1, 1], name='embeddings', filters=num_outputs_embedding, kernel_initializer=kernel_initializer, activation=embeddings_activation, data_format=data_format, is_training=is_training, padding=padding)
    return embeddings, embeddings_2


def network2d(input, is_training, num_outputs_embedding, actual_network, filters=64, levels=5, activation='relu', normalize=True, data_format='channels_first', padding='same'):
    if activation == 'selu':
        activation = tf.nn.selu
        kernel_initializer = selu_initializer
    elif activation == 'relu':
        activation = tf.nn.relu
        kernel_initializer = he_initializer
    elif activation == 'tanh':
        activation = tf.nn.tanh
        kernel_initializer = selu_initializer
    padding = padding
    embedding_axis = 1 if data_format == 'channels_first' else 4
    if normalize:
        embeddings_activation = lambda x, name: tf.nn.l2_normalize(x, dim=embedding_axis, name=name, epsilon=1e-4)
    else:
        if activation == tf.nn.selu:
            embeddings_activation = tf.nn.selu
        else:
            embeddings_activation = None

    embeddings_normalization = lambda x, name: tf.nn.l2_normalize(x, dim=embedding_axis, name=name, epsilon=1e-4)
    batch_size, channels, (num_frames, height, width) = get_batch_channel_image_size(input, data_format=data_format)

    with tf.variable_scope('unet_0'):
        unet = actual_network(num_filters_base=filters, kernel=[3, 3], num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet', padding=padding)
        unet_out = unet(input[:, 0, :, :, :], is_training)
        embeddings_2d = conv2d(unet_out, kernel_size=[1, 1], name='embeddings', filters=num_outputs_embedding * num_frames, kernel_initializer=kernel_initializer, activation=embeddings_activation, data_format=data_format, is_training=is_training, padding=padding)
        embeddings = tf.reshape(embeddings_2d, [batch_size, num_outputs_embedding, num_frames, height, width])
    with tf.variable_scope('unet_1'):
        normalized_embeddings = embeddings_normalization(embeddings, 'embeddings_normalized')
        normalized_embeddings_2d = tf.reshape(embeddings_2d, [batch_size, num_outputs_embedding * num_frames, height, width])
        input_concat = concat_channels([input[:, 0, :, :, :], normalized_embeddings_2d], name='input_concat', data_format=data_format)
        unet = actual_network(num_filters_base=filters, kernel=[3, 3], num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet', padding=padding)
        unet_out = unet(input_concat, is_training)
        embeddings_2_2d = conv2d(unet_out, kernel_size=[1, 1], name='embeddings', filters=num_outputs_embedding * num_frames, kernel_initializer=kernel_initializer, activation=embeddings_activation, data_format=data_format, is_training=is_training, padding=padding)
        embeddings_2 = tf.reshape(embeddings_2_2d, [batch_size, num_outputs_embedding, num_frames, height, width])
    return embeddings, embeddings_2

