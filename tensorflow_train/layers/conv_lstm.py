import tensorflow as tf
from tensorflow_train.layers.layers import conv2d, conv3d

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    """A LSTM cell with convolutions instead of multiplications.

    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalization=None, peephole=True, data_format='channels_last', reuse=None, is_training=False, dropout_ratio=0, name=''):
        super(ConvLSTMCell, self).__init__(_reuse=reuse, name=name)
        self._kernel = kernel
        self._filters = filters
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalization = normalization
        self._peephole = peephole
        self._data_format = data_format
        self._is_training = is_training
        self._dropout_ratio = dropout_ratio
        self._regularizer = tf.nn.l2_loss
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
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
        return self._size

    def call(self, x, state):
        c, h = state

        if self._dropout_ratio > 0:
            h = tf.layers.dropout(h, self._dropout_ratio, training=self._is_training)

        x = tf.concat([x, h], axis=self._feature_axis)
        n = x.shape[self._feature_axis].value
        m = 4 * self._filters if self._filters > 1 else 4
        W = tf.get_variable('kernel', self._kernel + [n, m], regularizer=self._regularizer)
        y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format_tf)
        if self._normalization is None:
            b = tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
            y = tf.nn.bias_add(y, b, data_format=self._data_format_tf)
        j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

        if self._peephole:
            i += tf.get_variable('W_ci', c.shape[1:], regularizer=self._regularizer) * c
            f += tf.get_variable('W_cf', c.shape[1:], regularizer=self._regularizer) * c

        if self._normalization is not None:
            j = self._normalization(j, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_j')
            i = self._normalization(i, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_i')
            f = self._normalization(f, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_f')

        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
            o += tf.get_variable('W_co', c.shape[1:], regularizer=self._regularizer) * c

        if self._normalization is not None:
            o = self._normalization(o, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_o')
            c = self._normalization(c, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_c')

        o = tf.sigmoid(o)
        h = o * self._activation(c)

        # TODO
        #tf.summary.histogram('forget_gate', f)
        #tf.summary.histogram('input_gate', i)
        #tf.summary.histogram('output_gate', o)
        #tf.summary.histogram('cell_state', c)

        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self, shape, filters, kernel, activation=tf.tanh, normalization=None, data_format='channels_last', reuse=None, is_training=False, name='', padding='same'):
        super(ConvGRUCell, self).__init__(_reuse=reuse, name=name)
        self._filters = filters
        self._kernel = kernel
        self._activation = activation
        self._normalization = normalization
        self._data_format = data_format
        self._is_training = is_training
        self._regularizer = tf.nn.l2_loss
        self._conv = conv2d
        self._padding = padding
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

    # def call(self, x, h):
    #     #channels = x.shape[self._feature_axis].value
    #
    #     with tf.variable_scope('gates'):
    #         inputs = tf.concat([x, h], axis=self._feature_axis)
    #         #n = channels + self._filters
    #         m = 2 * self._filters if self._filters > 1 else 2
    #         y = self._conv(inputs, m, self._kernel, data_format=self._data_format, name='kernel', padding='symmetric', is_training=self._is_training)
    #         r, u = tf.split(y, 2, axis=self._feature_axis)
    #         #W = tf.get_variable('kernel', self._kernel + [n, m], regularizer=self._regularizer)
    #         #y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format_tf)
    #         if self._normalization is not None:
    #             # r, u = tf.split(y, 2, axis=self._feature_axis)
    #             r = self._normalization(r, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_r')
    #             u = self._normalization(u, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_u')
    #         # else:
    #         #     b = tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    #         #     y = tf.nn.bias_add(y, b, data_format=self._data_format_tf)
    #         #     r, u = tf.split(y, 2, axis=self._feature_axis)
    #         r, u = tf.sigmoid(r), tf.sigmoid(u)
    #
    #         # TODO
    #         #tf.summary.histogram('reset_gate', r)
    #         #tf.summary.histogram('update_gate', u)
    #
    #     with tf.variable_scope('candidate'):
    #         inputs = tf.concat([x, r * h], axis=self._feature_axis)
    #         #n = channels + self._filters
    #         m = self._filters
    #         y = self._conv(inputs, m, self._kernel, data_format=self._data_format, name='kernel', padding='symmetric', is_training=self._is_training)
    #         #W = tf.get_variable('kernel', self._kernel + [n, m], regularizer=self._regularizer)
    #         #y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format_tf)
    #         if self._normalization is not None:
    #             y = self._normalization(y, is_training=self._is_training, data_format=self._data_format, name=self.name + '/norm_y')
    #         #else:
    #             #b = tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    #             #y = tf.nn.bias_add(y, b, data_format=self._data_format_tf)
    #         h = u * h + (1 - u) * self._activation(y)
    #
    #     return h, h


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
                           kernel_initializer=tf.constant_initializer(0))
            u = self._conv(inputs,
                           self._filters,
                           self._kernel,
                           data_format=self._data_format,
                           name='u',
                           padding=self._padding,
                           normalization=self._normalization,
                           activation=tf.sigmoid,
                           is_training=self._is_training,
                           kernel_initializer=tf.constant_initializer(0))

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
                           is_training=self._is_training)
            h = u * h + (1 - u) * y

        return h, h
