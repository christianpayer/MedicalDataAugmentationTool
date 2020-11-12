
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import concat_channels
from tensorflow_train.layers.conv_lstm import ConvLSTMCell, ConvGRUCell
from tensorflow_train.networks.unet_base import UnetBase, UnetBase2D
from tensorflow_train.utils.data_format import get_batch_channel_image_size

class UnetRecurrent(UnetBase):
    def recurrent(self, node, current_level, postfix, is_training):
        tf.add_to_collection('checkpoints', node)
        num_features = self.num_filters(current_level)
        batch_size, _, image_size = get_batch_channel_image_size(node, data_format=self.data_format)
        cell = self.recurrent_cell(image_size, num_features, postfix, is_training)
        if self.use_lstm_input_state:
            lstm_input_state = self.lstm_input_states[current_level]
        else:
            lstm_input_state = cell.zero_state(batch_size, tf.float32)
            self.lstm_input_states[current_level] = lstm_input_state
        node, lstm_output_state = cell(node, lstm_input_state)
        tf.add_to_collection('checkpoints', node)
        tf.add_to_collection('checkpoints', lstm_output_state)
        self.lstm_output_states[current_level] = lstm_output_state
        return node

    def recurrent_cell(self, shape, num_features, postfix, is_training):
        raise NotImplementedError

    def __call__(self, node, lstm_input_states, is_training):
        if lstm_input_states is None:
            print('Unet Recurrent with zero state')
            self.use_lstm_input_state = False
            self.lstm_output_states = [None] * self.num_levels
            self.lstm_input_states = [None] * self.num_levels
        else:
            print('Unet Recurrent with given state')
            self.use_lstm_input_state = True
            self.lstm_output_states = [None] * self.num_levels
            self.lstm_input_states = lstm_input_states
        return self.expanding(self.parallel(self.contracting(node, is_training), is_training), is_training)

    def parallel_and_expanding_with_input_states(self, level_nodes, lstm_input_states, is_training):
        if lstm_input_states is None:
            print('Unet Recurrent with zero state')
            self.use_lstm_input_state = False
            self.lstm_output_states = [None] * self.num_levels
            self.lstm_input_states = [None] * self.num_levels
        else:
            print('Unet Recurrent with given state')
            self.use_lstm_input_state = True
            self.lstm_output_states = [None] * self.num_levels
            self.lstm_input_states = lstm_input_states
        return self.expanding(self.parallel(level_nodes, is_training), is_training)


class UnetLstm2D(UnetRecurrent, UnetBase2D):
    def recurrent_cell(self, shape, num_features, postfix, is_training):
        return ConvLSTMCell(shape, num_features, [3, 3], data_format=self.data_format, normalize=False, peephole=True, name='lstm' + postfix)

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.recurrent(node, current_level, '', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        #node = self.conv(node, current_level, '1', is_training)
        return node


class UnetLstmAll2D(UnetLstm2D):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return parallel_node + upsample_node

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        node = self.recurrent(node, current_level, '_lstm', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        node = self.recurrent(node, current_level, '_lstm', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        node = self.recurrent(node, current_level, '_lstm', is_training)
        return node


class UnetGru2D(UnetRecurrent, UnetBase2D):
    def recurrent_cell(self, shape, num_features, postfix, is_training):
        return ConvGRUCell(shape, num_features, [3, 3], activation=tf.nn.relu, data_format=self.data_format, normalize=False, name='gru' + postfix)

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        #node = self.recurrent(node, current_level, '', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        node = self.recurrent(node, current_level, '', is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        #node = self.recurrent(node, current_level, '', is_training)
        return node
