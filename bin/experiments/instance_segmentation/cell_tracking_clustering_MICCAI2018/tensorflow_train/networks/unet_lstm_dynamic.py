
import tensorflow as tf
from tensorflow_train.layers.conv_lstm import ConvGRUCell
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.layers.layers import conv2d, add, max_pool2d, upsample2d


class UnetRecurrentWithStates(UnetBase):
    def __init__(self, shape, **kwargs):
        super(UnetRecurrentWithStates, self).__init__(**kwargs)
        self.shape = shape
        if self.data_format == 'channels_last':
            self.state_size = tuple([tf.TensorShape([s / (2 ** i) for s in shape] + [self.num_filters(i)]) for i in range(self.num_levels)])
        else:
            self.state_size = tuple([tf.TensorShape([self.num_filters(i)] + [s / (2 ** i) for s in shape]) for i in range(self.num_levels)])

    def recurrent(self, node, current_level, postfix, is_training):
        num_features = self.num_filters(current_level)
        batch_size, _, image_size = get_batch_channel_image_size(node, data_format=self.data_format)
        cell = self.recurrent_cell(image_size, num_features, postfix, is_training)
        lstm_input_state = self.lstm_input_states[current_level]
        node, lstm_output_state = cell(node, lstm_input_state)
        self.lstm_output_states[current_level] = lstm_output_state
        return node

    def recurrent_cell(self, shape, num_features, postfix, is_training):
        raise NotImplementedError

    def __call__(self, node, lstm_input_states, is_training):
        print('Unet Recurrent with given state')
        self.lstm_output_states = [None] * self.num_levels
        self.lstm_input_states = lstm_input_states
        return self.expanding(self.parallel(self.contracting(node, is_training), is_training), is_training), self.lstm_output_states


class UnetRecurrentCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, shape, unet_recurrent, kernel, num_outputs, activation=None, data_format='channels_first', reuse=None, is_training=False, name='', padding='same'):
        super(UnetRecurrentCell, self).__init__(_reuse=reuse, name=name)
        self.kernel = kernel
        self.num_outputs = num_outputs
        self.unet_recurrent = unet_recurrent
        self.activation = activation
        self.data_format = data_format
        self.is_training = is_training
        self.padding = padding
        if data_format == 'channels_last':
            self.output_size_internal = tf.TensorShape(shape + [self.num_outputs])
        elif data_format == 'channels_first':
            self.output_size_internal = tf.TensorShape([self.num_outputs] + shape)
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return self.unet_recurrent.state_size

    @property
    def output_size(self):
        return self.output_size_internal

    def call(self, node, states):
        node, output_states = self.unet_recurrent(node, list(states), self.is_training)
        return node, tuple(output_states)


# 2D classes

class UnetGruWithStates2D(UnetRecurrentWithStates):
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
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format,
                      padding=self.padding)

    def recurrent_cell(self, shape, num_features, postfix, is_training):
        return ConvGRUCell(shape,
                           num_features,
                           [3, 3],
                           activation=tf.nn.relu,
                           data_format=self.data_format,
                           normalization=None,
                           name='gru' + postfix,
                           is_training=is_training,
                           padding=self.padding)

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        node = add([parallel_node, upsample_node], name='add' + str(current_level))
        return node

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.recurrent(node, current_level, '', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return node


class UnetRecurrentCell2D(UnetRecurrentCell):
    def call(self, node, states):
        node, output_states = self.unet_recurrent(node, list(states), self.is_training)
        node = conv2d(node,
                      self.num_outputs,
                      self.kernel,
                      'output',
                      data_format=self.data_format,
                      padding=self.padding,
                      activation=self.activation,
                      is_training=self.is_training)
        return node, tuple(output_states)
