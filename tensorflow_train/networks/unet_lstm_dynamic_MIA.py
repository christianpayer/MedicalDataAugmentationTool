
import tensorflow as tf
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.layers.layers import conv2d


class UnetRecurrentWithStates(UnetBase):
    def __init__(self, shape, recurrents_per_level=1, lstm=False, **kwargs):
        super(UnetRecurrentWithStates, self).__init__(**kwargs)
        self.shape = shape
        self.recurrents_per_level = recurrents_per_level
        state_size_list = []
        for i in range(self.num_levels):
            for j in range(self.recurrents_per_level):
                if self.data_format == 'channels_last':
                    state_size = tf.TensorShape([s / (2 ** i) for s in shape] + [self.num_filters(i)])
                else:
                    state_size = tf.TensorShape([self.num_filters(i)] + [s / (2 ** i) for s in shape])
                if lstm:
                    state_size_list.append(tf.nn.rnn_cell.LSTMStateTuple(state_size, state_size))
                else:
                    state_size_list.append(state_size)
        self.state_size = tuple(state_size_list)

    def recurrent(self, node, current_level, index_on_level, postfix, is_training):
        num_features = self.num_filters(current_level)
        batch_size, channel_size, image_size = get_batch_channel_image_size(node, data_format=self.data_format)
        cell = self.recurrent_cell([channel_size] + image_size, num_features, postfix, is_training)
        lstm_input_state = self.lstm_input_states[current_level * self.recurrents_per_level + index_on_level]
        node, lstm_output_state = cell(node, lstm_input_state)
        self.lstm_output_states[current_level * self.recurrents_per_level + index_on_level] = lstm_output_state
        return node

    def recurrent_cell(self, shape, num_features, postfix, is_training):
        raise NotImplementedError

    def __call__(self, node, lstm_input_states, is_training):
        print('Unet Recurrent with given state')
        self.lstm_output_states = [None] * (self.num_levels * self.recurrents_per_level)
        self.lstm_input_states = lstm_input_states
        return self.expanding(self.parallel(self.contracting(node, is_training), is_training), is_training), self.lstm_output_states


class UnetRecurrentCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, shape, unet_recurrent, kernel, num_outputs_first_conv, num_outputs, kernel_initializer, input_activation=None, output_activation=None, data_format='channels_first', reuse=None, is_training=False, name='', padding='same'):
        super(UnetRecurrentCell, self).__init__(_reuse=reuse, name=name)
        self.kernel = kernel
        self.num_outputs_first_conv = num_outputs_first_conv
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        self.unet_recurrent = unet_recurrent
        self.input_activation = input_activation
        self.output_activation = output_activation
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


class UnetRecurrentCell2D(UnetRecurrentCell):
    def call(self, node, states):
        node = conv2d(node,
                      self.num_outputs_first_conv,
                      [1, 1],
                      'input',
                      data_format=self.data_format,
                      padding=self.padding,
                      kernel_initializer=self.kernel_initializer,
                      activation=self.input_activation,
                      is_training=self.is_training)
        node, output_states = self.unet_recurrent(node, list(states), self.is_training)
        node = conv2d(node,
                      self.num_outputs,
                      [1, 1],
                      'output',
                      data_format=self.data_format,
                      padding=self.padding,
                      kernel_initializer=self.kernel_initializer,
                      activation=self.output_activation,
                      is_training=self.is_training)
        return node, tuple(output_states)
