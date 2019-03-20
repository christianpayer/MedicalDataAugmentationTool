
import numpy as np
import tensorflow as tf
from tensorflow_train.layers.initializers import he_initializer, zeros_initializer
from tensorflow_train.utils.data_format import get_channel_index
from tensorflow_train.utils.print_utils import print_conv_parameters, print_pool_parameters, print_dropout_parameters, print_upsample_parameters, print_shape_parameters, print_dense_parameters


debug_print_conv = True
debug_print_dense = True
debug_print_pool = False
debug_print_upsample = False
debug_print_others = False


def pad_for_conv(inputs, kernel_size, name, padding, data_format):
    if padding in ['symmetric', 'reflect']:
        # TODO check if this works for even kernels
        channel_index = get_channel_index(inputs, data_format)
        paddings = np.array([[0, 0]] + [[int(ks / 2)] * 2 for ks in kernel_size])
        paddings = np.insert(paddings, channel_index, [0, 0], axis=0)
        outputs = tf.pad(inputs, paddings, mode=padding, name=name+'/pad')
        padding_for_conv = 'valid'
    else:
        outputs = inputs
        padding_for_conv = padding
    return outputs, padding_for_conv


def conv2d(inputs,
           filters,
           kernel_size,
           name,
           activation=None,
           kernel_initializer=he_initializer,
           bias_initializer=zeros_initializer,
           normalization=None,
           is_training=False,
           data_format='channels_first',
           padding='same',
           strides=(1, 1),
           debug_print=debug_print_conv):
    node, padding_for_conv = pad_for_conv(inputs=inputs,
                                          kernel_size=kernel_size,
                                          name=name,
                                          padding=padding,
                                          data_format=data_format)
    outputs = tf.layers.conv2d(inputs=node,
                               filters=filters,
                               kernel_size=kernel_size,
                               name=name,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               trainable=is_training,
                               data_format=data_format,
                               kernel_regularizer=tf.nn.l2_loss,
                               padding=padding_for_conv,
                               strides=strides)

    if normalization is not None:
        outputs = normalization(outputs, is_training=is_training, data_format=data_format, name=name+'/norm')

    if activation is not None:
        outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=strides)

    return outputs


def conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     name,
                     activation=None,
                     kernel_initializer=he_initializer,
                     bias_initializer=zeros_initializer,
                     normalization=None,
                     is_training=False,
                     data_format='channels_first',
                     padding='same',
                     strides=(1, 1),
                     debug_print=debug_print_conv):
    outputs = tf.layers.conv2d_transpose(inputs=inputs,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         name=name,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         trainable=is_training,
                                         data_format=data_format,
                                         kernel_regularizer=tf.nn.l2_loss,
                                         padding=padding,
                                         strides=strides)

    if normalization is not None:
        outputs = normalization(outputs, is_training=is_training, data_format=data_format, name=name+'/norm')

    if activation is not None:
        outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=strides)

    return outputs


def upsample2d(inputs, kernel_size, name='', data_format='channels_first', debug_print=debug_print_upsample):
    outputs = tf.contrib.keras.layers.UpSampling2D(kernel_size, data_format=data_format, name=name)(inputs)
    if debug_print:
        print_upsample_parameters('nn',
                                  inputs,
                                  outputs,
                                  kernel_size,
                                  name,
                                  data_format,
                                  'same',
                                  kernel_size)
    return outputs


def avg_pool2d(inputs, kernel_size, strides=None, name='', padding='same', data_format='channels_first', debug_print=debug_print_pool):
    if strides is None:
        strides = kernel_size
    outputs = tf.layers.average_pooling2d(inputs, kernel_size, strides, padding='same', data_format=data_format, name=name)
    if debug_print:
        print_pool_parameters(pool_type='avg',
                              inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=name,
                              data_format=data_format,
                              padding=padding)
    return outputs


def max_pool2d(inputs, kernel_size, strides=None, name='', padding='same', data_format='channels_first', debug_print=debug_print_pool):
    if strides is None:
        strides = kernel_size
    outputs = tf.layers.max_pooling2d(inputs, kernel_size, strides, padding='same', data_format=data_format, name=name)
    if debug_print:
        print_pool_parameters(pool_type='avg',
                              inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=name,
                              data_format=data_format,
                              padding=padding)
    return outputs


def conv3d(inputs,
           filters,
           kernel_size,
           name,
           activation=None,
           kernel_initializer=he_initializer,
           bias_initializer=zeros_initializer,
           normalization=None,
           is_training=False,
           data_format='channels_first',
           padding='same',
           strides=(1, 1, 1),
           debug_print=debug_print_conv):
    node, padding_for_conv = pad_for_conv(inputs=inputs,
                                          kernel_size=kernel_size,
                                          name=name,
                                          padding=padding,
                                          data_format=data_format)
    outputs = tf.layers.conv3d(inputs=node,
                               filters=filters,
                               kernel_size=kernel_size,
                               name=name,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               trainable=is_training,
                               data_format=data_format,
                               kernel_regularizer=tf.nn.l2_loss,
                               padding=padding_for_conv)

    if normalization is not None:
        outputs = normalization(outputs, is_training=is_training, data_format=data_format, name=name+'/norm')

    if activation is not None:
        outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=strides)

    return outputs


def upsample3d(inputs, kernel_size, name='', data_format='channels_first', debug_print=debug_print_upsample):
    outputs = tf.contrib.keras.layers.UpSampling3D(kernel_size, data_format=data_format, name=name)(inputs)
    if debug_print:
        print_upsample_parameters('nn',
                                  inputs,
                                  outputs,
                                  kernel_size,
                                  name,
                                  data_format,
                                  'same',
                                  kernel_size)
    return outputs


def avg_pool3d(inputs, kernel_size, strides=None, name='', padding='same', data_format='channels_first', debug_print=debug_print_pool):
    if strides is None:
        strides = kernel_size
    outputs = tf.layers.average_pooling3d(inputs, kernel_size, strides, padding='same', data_format=data_format, name=name)
    if debug_print:
        print_pool_parameters(pool_type='avg',
                              inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=name,
                              data_format=data_format,
                              padding=padding)
    return outputs


def max_pool3d(inputs, kernel_size, strides=None, name='', padding='same', data_format='channels_first', debug_print=debug_print_pool):
    if strides is None:
        strides = kernel_size
    outputs = tf.layers.max_pooling3d(inputs, kernel_size, strides, padding=padding, data_format=data_format, name=name)
    if debug_print:
        print_pool_parameters(pool_type='max',
                              inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=name,
                              data_format=data_format,
                              padding=padding)
    return outputs


def concat_channels(inputs, name='', data_format='channels_first', debug_print=debug_print_others):
    axis = get_channel_index(inputs[0], data_format)
    outputs = tf.concat(inputs, axis=axis, name=name)
    if debug_print:
        print_shape_parameters(inputs, outputs, name, 'concat')
    return outputs


def add(inputs, name='', debug_print=debug_print_others):
    outputs = tf.add_n(inputs, name=name)
    if debug_print:
        print_shape_parameters(inputs[0], outputs, name, 'add')
    return outputs


def mult(input0, input1, name='', debug_print=debug_print_others):
    outputs = tf.multiply(input0, input1, name=name)
    if debug_print:
        print_shape_parameters(input0, outputs, name, 'mult')
    return outputs


def flatten(inputs, name='', debug_print=debug_print_others):
    outputs = tf.layers.flatten(inputs, name)
    if debug_print:
        print_shape_parameters(inputs, outputs, name, 'flatten')
    return outputs


def dropout(inputs, rate, name='', is_training=False, debug_print=debug_print_others):
    outputs = tf.layers.dropout(inputs, rate=rate, name=name, training=is_training)
    if debug_print:
        print_dropout_parameters(name=name, rate=rate, is_training=is_training)
    return outputs


def dense(inputs,
          units,
          name,
          activation=None,
          kernel_initializer=he_initializer,
          bias_initializer=zeros_initializer,
          normalization=None,
          is_training=False,
          debug_print=debug_print_dense):
    outputs = tf.layers.dense(inputs=inputs,
                              units=units,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=tf.nn.l2_loss,
                              name=name,
                              bias_initializer=bias_initializer)

    if normalization is not None:
        outputs = normalization(outputs, is_training=is_training, name=name+'/norm')

    if activation is not None:
        outputs = activation(outputs)

    if debug_print:
        print_dense_parameters(inputs=inputs,
                               outputs=outputs,
                               name=name,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               normalization=normalization,
                               is_training=is_training)

    return outputs
