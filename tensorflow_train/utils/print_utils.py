
import tensorflow as tf

from tensorflow_train.layers.initializers import he_initializer, selu_initializer, zeros_initializer
from tensorflow_train.layers.normalizers import batch_norm, instance_norm, layer_norm, batch_norm_dense


def print_tensor_shape(node):
    print(node.get_shape().as_list())


def printable_activation(activation):
    if activation == tf.nn.relu:
        return 'relu'
    elif activation == tf.nn.sigmoid:
        return 'sig'
    elif activation == tf.nn.tanh:
        return 'tanh'
    elif activation == tf.nn.selu:
        return 'selu'
    elif activation == tf.nn.softplus:
        return 'softplus'
    else:
        return activation


def printable_initializer(initializer):
    if initializer == he_initializer:
        return 'he'
    elif initializer == selu_initializer:
        return 'selu'
    elif initializer == zeros_initializer:
        return '0'
    elif isinstance(initializer, tf.initializers.variance_scaling):
        return 'var({},{},{})'.format(initializer.scale, initializer.mode, initializer.distribution)
    elif isinstance(initializer, tf.initializers.truncated_normal):
        return 'norm({}±{})'.format(initializer.mean, initializer.stddev)
    elif isinstance(initializer, tf.initializers.constant):
        return '{}'.format(initializer.value)
    else:
        return initializer


def printable_normalization(normalization):
    if normalization == batch_norm:
        return 'batch'
    elif normalization == instance_norm:
        return 'instance'
    elif normalization == layer_norm:
        return 'layer'
    elif normalization == batch_norm_dense:
        return 'batch'
    else:
        return normalization


def print_conv_parameters(inputs,
                          outputs,
                          kernel_size,
                          name,
                          activation,
                          kernel_initializer,
                          bias_initializer,
                          normalization,
                          is_training,
                          data_format,
                          padding,
                          strides):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: ' \
                   'in={} ' \
                   'out={} ' \
                   'ks={} ' \
                   's={} '\
                   'pad={} ' \
                   'act={} ' \
                   'k_init={} ' \
                   'b_init={} ' \
                   'norm={} ' \
                   'train={} ' \
                   'format={}' \
        .format(name,
                inputs_shape,
                outputs_shape,
                kernel_size,
                strides,
                padding,
                printable_activation(activation),
                printable_initializer(kernel_initializer),
                printable_initializer(bias_initializer),
                printable_normalization(normalization),
                is_training,
                data_format)
    print(print_string)


def print_dense_parameters(inputs,
                           outputs,
                           name,
                           activation,
                           kernel_initializer,
                           bias_initializer,
                           normalization,
                           is_training):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: ' \
                   'in={} ' \
                   'out={} ' \
                   'act={} ' \
                   'k_init={} ' \
                   'b_init={} ' \
                   'norm={} ' \
                   'train={}' \
        .format(name,
                inputs_shape,
                outputs_shape,
                printable_activation(activation),
                printable_initializer(kernel_initializer),
                printable_initializer(bias_initializer),
                printable_normalization(normalization),
                is_training)
    print(print_string)


def print_pool_parameters(pool_type,
                          inputs,
                          outputs,
                          kernel_size,
                          name,
                          data_format,
                          padding,
                          strides):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: ' \
                   'type={} ' \
                   'in={} ' \
                   'out={} ' \
                   'ks={} ' \
                   's={} ' \
                   'pad={} ' \
                   'format={}' \
        .format(name,
                pool_type,
                inputs_shape,
                outputs_shape,
                kernel_size,
                strides,
                padding,
                data_format)
    print(print_string)


def print_upsample_parameters(upsample_type,
                              inputs,
                              outputs,
                              kernel_size,
                              name,
                              data_format,
                              padding,
                              strides):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: ' \
                   'type={} ' \
                   'in={} ' \
                   'out={} ' \
                   'ks={} ' \
                   's={} ' \
                   'pad={} ' \
                   'format={}' \
        .format(name,
                upsample_type,
                inputs_shape,
                outputs_shape,
                kernel_size,
                strides,
                padding,
                data_format)
    print(print_string)


def print_dropout_parameters(name, rate, is_training):
    print_string = '{}: rate={} train={}'.format(name, rate, is_training)
    print(print_string)


def print_shape_parameters(inputs,
                           outputs,
                           name,
                           type):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: type={} in={} out={}'.format(name, type, inputs_shape, outputs_shape)
    print(print_string)
