import tensorflow as tf
from tensorflow_train.layers.layers import conv2d, add, max_pool2d, upsample2d
from tensorflow_train.layers.initializers import selu_initializer, he_initializer
from tensorflow_train.networks.unet_lstm_dynamic_MIA import UnetRecurrentCell2D, UnetRecurrentWithStates


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
            self._feature_axis = self._size.ndims
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


def network(input, is_training, num_outputs_embedding, stacked_hourglass=True, filters=64, levels=7, activation='relu', normalize=False, data_format='channels_first', padding='same', parallel_iterations=8, actual_network=None):
    if activation == 'selu':
        activation = tf.nn.selu
        kernel_initializer = selu_initializer
    elif activation == 'relu':
        activation = tf.nn.relu
        kernel_initializer = he_initializer
    elif activation == 'tanh':
        activation = tf.nn.tanh
        kernel_initializer = selu_initializer

    if data_format == 'channels_first':
        image_size = input.get_shape().as_list()[3:5]
        input_transposed = tf.transpose(input, [0, 2, 1, 3, 4])
        embedding_axis = 2
    else:
        image_size = input.get_shape().as_list()[2:4]
        input_transposed = input
        embedding_axis = 4

    if normalize:
        embeddings_activation = lambda x, name: tf.nn.l2_normalize(x, axis=embedding_axis, name=name, epsilon=1e-4)
    else:
        if activation == tf.nn.selu:
            embeddings_activation = tf.nn.selu
        else:
            embeddings_activation = None
    normalization = None
    embeddings_normalization = lambda x, name: tf.nn.l2_normalize(x, axis=embedding_axis, name=name, epsilon=1e-4)

    unet_recurrent_0 = actual_network(normalization=normalization, shape=image_size, num_filters_base=filters, num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet_0', padding=padding)
    unet_recurrent_cell_0 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_0, shape=image_size, num_outputs_first_conv=filters, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, kernel_initializer=kernel_initializer, input_activation=activation, output_activation=embeddings_activation, is_training=is_training, name='unet_0_cell', padding=padding)
    embeddings_transposed, states = tf.nn.dynamic_rnn(unet_recurrent_cell_0, input_transposed, dtype=input.dtype, swap_memory=True, parallel_iterations=parallel_iterations)

    embeddings = tf.transpose(embeddings_transposed, [0, 2, 1, 3, 4])

    if stacked_hourglass:
        if not normalize:
            embeddings_transposed_normalized = embeddings_normalization(embeddings_transposed, name='embeddings_normalization')
        else:
            embeddings_transposed_normalized = embeddings_transposed
        input_lstm_1 = tf.concat([input_transposed, embeddings_transposed_normalized], name='embedding_input_concat', axis=2)
        unet_recurrent_1 = actual_network(normalization=normalization, shape=image_size, num_filters_base=filters, num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet_1', padding=padding)
        unet_recurrent_cell_1 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_1, shape=image_size, num_outputs_first_conv=filters, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, kernel_initializer=kernel_initializer, input_activation=activation, output_activation=embeddings_activation, is_training=is_training, name='unet_1_cell', padding=padding)
        embeddings_transposed_2, states_2 = tf.nn.dynamic_rnn(unet_recurrent_cell_1, input_lstm_1, dtype=input.dtype, swap_memory=True, parallel_iterations=parallel_iterations)
        embeddings_2 = tf.transpose(embeddings_transposed_2, [0, 2, 1, 3, 4])
        return embeddings, embeddings_2
    else:
        return embeddings


def network_single_frame_with_lstm_states(input, num_outputs_embedding, stacked_hourglass=True, filters=64, levels=7, activation='relu', normalize=False, padding='same', data_format='channels_first', parallel_iterations=None, actual_network=None):
    if activation == 'selu':
        activation = tf.nn.selu
        kernel_initializer = selu_initializer
    elif activation == 'relu':
        activation = tf.nn.relu
        kernel_initializer = he_initializer
    elif activation == 'tanh':
        activation = tf.nn.tanh
        kernel_initializer = selu_initializer

    if data_format == 'channels_first':
        image_size = input.get_shape().as_list()[2:4]
        embedding_axis = 1
    else:
        image_size = input.get_shape().as_list()[1:3]
        embedding_axis = 3

    if normalize:
        embeddings_activation = lambda x, name: tf.nn.l2_normalize(x, axis=embedding_axis, name=name, epsilon=1e-4)
    else:
        if activation == tf.nn.selu:
            embeddings_activation = tf.nn.selu
        else:
            embeddings_activation = None
    normalization = None
    embeddings_normalization = lambda x, name: tf.nn.l2_normalize(x, axis=embedding_axis, name=name, epsilon=1e-4)

    is_training = False
    unet_recurrent_0 = actual_network(normalization=normalization, shape=image_size, num_filters_base=filters, num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet_0', padding=padding)
    unet_recurrent_cell_0 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_0, shape=image_size, num_outputs_first_conv=filters, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, kernel_initializer=kernel_initializer, input_activation=activation, output_activation=embeddings_activation, is_training=is_training, name='unet_0_cell', padding=padding)
    lstm_input_states = unet_recurrent_cell_0.zero_state(1, input.dtype)
    embeddings, states = unet_recurrent_cell_0(input, lstm_input_states)
    if stacked_hourglass:
        if not normalize:
            embeddings_normalized = embeddings_normalization(embeddings, name='embeddings_normalization')
        else:
            embeddings_normalized = embeddings
        input_lstm_1 = tf.concat([input, embeddings_normalized], name='embedding_input_concat', axis=1)
        unet_recurrent_1 = actual_network(normalization=normalization, shape=image_size, num_filters_base=filters, num_levels=levels, data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training, name='unet_1', padding=padding)
        unet_recurrent_cell_1 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_1, shape=image_size, num_outputs_first_conv=filters, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, kernel_initializer=kernel_initializer, input_activation=activation, output_activation=embeddings_activation, is_training=is_training, name='unet_1_cell', padding=padding)
        lstm_input_states_1 = unet_recurrent_cell_1.zero_state(1, input.dtype)
        embeddings_2, states_2 = unet_recurrent_cell_1(input_lstm_1, lstm_input_states_1)
        lstm_input_states = lstm_input_states + lstm_input_states_1
        lstm_output_states = states + states_2
        return lstm_input_states, lstm_output_states, embeddings, embeddings_2  # , embeddings_mean_shift
    else:
        lstm_output_states = states
        return lstm_input_states, lstm_output_states, embeddings

