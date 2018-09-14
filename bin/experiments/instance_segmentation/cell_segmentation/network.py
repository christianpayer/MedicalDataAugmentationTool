
import tensorflow as tf
from tensorflow_train.layers.layers import conv2d, concat_channels
from tensorflow_train.networks.unet import UnetParallel2D
from tensorflow_train.layers.initializers import he_initializer


def network(input, is_training, num_outputs_embedding, data_format='channels_first'):
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    with tf.variable_scope('unet_0'):
        unet = UnetParallel2D(num_filters_base=64, kernel=[3, 3], num_levels=7, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet')
        embeddings = conv2d(unet(input, is_training), kernel_size=[3, 3], name='embeddings', filters=num_outputs_embedding, kernel_initializer=kernel_initializer, activation=None, data_format=data_format)
    with tf.variable_scope('unet_1'):
        embeddings_normalized = tf.nn.l2_normalize(embeddings, axis=1)
        input_concat = concat_channels([input, embeddings_normalized], name='input_concat', data_format=data_format)
        unet = UnetParallel2D(num_filters_base=64, kernel=[3, 3], num_levels=7, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet')
        embeddings_2 = conv2d(unet(input_concat, is_training), kernel_size=[3, 3], name='embeddings', filters=num_outputs_embedding, kernel_initializer=kernel_initializer, activation=None, data_format=data_format)
    return embeddings, embeddings_2
