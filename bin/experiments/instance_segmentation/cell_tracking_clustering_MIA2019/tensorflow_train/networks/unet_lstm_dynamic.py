import tensorflow as tf
from tensorflow_train.layers.layers import conv2d, add, max_pool2d, upsample2d
from tensorflow_train.layers.initializers import selu_initializer, he_initializer
from tensorflow_train.utils.tensorflow_util import masked_apply
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.networks.unet_base import UnetBase

class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""
    def __init__(self, shape, filters, kernel, activation=tf.tanh, kernel_initializer=selu_initializer, normalization=None, data_format='channels_last', reuse=None, is_training=False, name='', padding='same', factor_bias_initializer=tf.constant_initializer(0)):
        super(ConvGRUCell, self).__init__(_reuse=reuse, name=name)
        self._filters = filters
        self._kernel = kernel
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._normalization = normalization
        self._data_format = data_format
        self._is_training = is_training
        self._regularizer = tf.nn.l2_loss
        self._conv = conv2d
        self._padding = padding
        self.factor_bias_initializer = factor_bias_initializer
        if data_format == 'channels_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims - 1
            self._data_format_tf = 'NHWC'
        elif data_format == 'channels_first':
            self._size = tf.TensorShape([self._filters] + shape)
            self._feature_axis = 1
            self._data_format_tf = 'NCHW'
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def call(self, x, h):
        with tf.variable_scope('gates'):
            inputs = tf.concat([x, h], axis=self._feature_axis)
            r = self._conv(inputs,
                           self._filters,
                           self._kernel,
                           data_format=self._data_format,
                           name='r',
                           padding=self._padding,
                           normalization=self._normalization,
                           activation=tf.sigmoid,
                           is_training=self._is_training,
                           kernel_initializer=selu_initializer,
                           bias_initializer=self.factor_bias_initializer)
            u = self._conv(inputs,
                           self._filters,
                           self._kernel,
                           data_format=self._data_format,
                           name='u',
                           padding=self._padding,
                           normalization=self._normalization,
                           activation=tf.sigmoid,
                           is_training=self._is_training,
                           kernel_initializer=selu_initializer,
                           bias_initializer=self.factor_bias_initializer)

        with tf.variable_scope('candidate'):
            inputs = tf.concat([x, r * h], axis=self._feature_axis)
            y = self._conv(inputs,
                           self._filters,
                           self._kernel,
                           data_format=self._data_format,
                           name='y',
                           padding=self._padding,
                           normalization=self._normalization,
                           activation=self._activation,
                           is_training=self._is_training,
                           kernel_initializer=self._kernel_initializer)
            h = u * h + (1 - u) * y

        return h, h


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


class UnetIntermediateGruWithStates2D(UnetRecurrentWithStates):
    def __init__(self, *args, **kwargs):
        super(UnetIntermediateGruWithStates2D, self).__init__(recurrents_per_level=1, lstm=False, *args, **kwargs)

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

    def recurrent_cell(self, shape, num_features, postfix, is_training):
        return ConvGRUCell(shape,
                           num_features,
                           [3, 3],
                           activation=self.activation,
                           kernel_initializer=self.kernel_initializer,
                           data_format=self.data_format,
                           normalization=self.normalization,
                           name='gru' + postfix,
                           is_training=is_training,
                           padding=self.padding,
                           factor_bias_initializer=tf.constant_initializer(0))

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        node = add([parallel_node, upsample_node], name='add' + str(current_level))
        return node

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.recurrent(node, current_level, 0, '0', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        return node


def network_single_frame_with_lstm_states(input, num_outputs_embedding, filters=64, levels=7, padding='same', data_format='channels_first'):
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    if data_format == 'channels_first':
        image_size = input.get_shape().as_list()[2:4]
        embedding_axis = 1
    else:
        image_size = input.get_shape().as_list()[1:3]
        embedding_axis = 3
    print(input)

    embeddings_activation = None
    normalization = None
    embeddings_normalization = lambda x, name: tf.nn.l2_normalize(x, dim=embedding_axis, name=name, epsilon=1e-4)

    is_training = False
    unet_recurrent_0 = UnetIntermediateGruWithStates2D(normalization=normalization, shape=image_size, num_filters_base=filters, num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet_0', padding=padding)
    unet_recurrent_cell_0 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_0, shape=image_size, num_outputs_first_conv=filters, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, kernel_initializer=kernel_initializer, input_activation=activation, output_activation=embeddings_activation, is_training=is_training, name='unet_0_cell', padding=padding)
    lstm_input_states_0 = unet_recurrent_cell_0.zero_state(1, input.dtype)
    embeddings_0, lstm_output_states_0 = unet_recurrent_cell_0(input, lstm_input_states_0)
    embeddings_0_normalized = embeddings_normalization(embeddings_0, name='embeddings_normalization')
    input_unet_lstm_1 = tf.concat([input, embeddings_0_normalized], name='embedding_input_concat', axis=embedding_axis)
    unet_recurrent_1 = UnetIntermediateGruWithStates2D(normalization=normalization, shape=image_size, num_filters_base=filters, num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet_1', padding=padding)
    unet_recurrent_cell_1 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_1, shape=image_size, num_outputs_first_conv=filters, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, kernel_initializer=kernel_initializer, input_activation=activation, output_activation=embeddings_activation, is_training=is_training, name='unet_1_cell', padding=padding)
    lstm_input_states_1 = unet_recurrent_cell_1.zero_state(1, input.dtype)
    embeddings_1, lstm_output_states_1 = unet_recurrent_cell_1(input_unet_lstm_1, lstm_input_states_1)
    lstm_input_states = lstm_input_states_0 + lstm_input_states_1
    lstm_output_states = lstm_output_states_0 + lstm_output_states_1

    return lstm_input_states, lstm_output_states, embeddings_0, embeddings_1

