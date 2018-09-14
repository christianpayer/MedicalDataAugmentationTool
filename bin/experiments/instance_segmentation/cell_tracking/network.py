
import tensorflow as tf
from tensorflow_train.networks.unet_lstm_dynamic import UnetGruWithStates2D, UnetRecurrentCell2D


def network(input, is_training, num_outputs_embedding, data_format='channels_first', parallel_iterations=8):
    if data_format == 'channels_first':
        image_size = input.get_shape().as_list()[3:5]
        input_transposed = tf.transpose(input, [0, 2, 1, 3, 4])
    else:
        image_size = input.get_shape().as_list()[2:4]
        input_transposed = input

    normalization = None
    num_levels = 7
    num_filters_base = 64
    num_filters_base_2 = 64
    padding = 'same'
    unet_recurrent_0 = UnetGruWithStates2D(shape=image_size, num_filters_base=num_filters_base, num_levels=num_levels, data_format=data_format, normalization=normalization, is_training=is_training, name='unet_0', padding=padding)
    unet_recurrent_cell_0 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_0, shape=image_size, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, is_training=is_training, name='unet_0_cell', padding=padding)
    embeddings_transposed, states = tf.nn.dynamic_rnn(unet_recurrent_cell_0, input_transposed, dtype=tf.float32, swap_memory=True, parallel_iterations=parallel_iterations)
    #embeddings = tf.transpose(embeddings_transposed, [0, 2, 1, 3, 4])
    embeddings_transposed_normalized = tf.nn.l2_normalize(embeddings_transposed, dim=2)

    input_lstm_1 = tf.concat([input_transposed, embeddings_transposed_normalized], name='embedding_input_concat', axis=2)
    unet_recurrent_1 = UnetGruWithStates2D(shape=image_size, num_filters_base=num_filters_base_2, num_levels=num_levels, data_format=data_format, normalization=normalization, is_training=is_training, name='unet_1', padding=padding)
    unet_recurrent_cell_1 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_1, shape=image_size, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, is_training=is_training, name='unet_1_cell', padding=padding)
    embeddings_transposed_2, states_2 = tf.nn.dynamic_rnn(unet_recurrent_cell_1, input_lstm_1, dtype=tf.float32, swap_memory=True, parallel_iterations=parallel_iterations)
    embeddings = tf.transpose(embeddings_transposed, [0, 2, 1, 3, 4])
    embeddings_2 = tf.transpose(embeddings_transposed_2, [0, 2, 1, 3, 4])

    return embeddings, embeddings_2


def network_single_frame_with_lstm_states(input, num_outputs_embedding, data_format='channels_first'):
    if data_format == 'channels_first':
        image_size = input.get_shape().as_list()[2:4]
    else:
        image_size = input.get_shape().as_list()[1:3]

    normalization = None
    is_training = False
    num_levels = 7
    num_filters_base = 64
    num_filters_base_2 = 64
    padding = 'same'
    unet_recurrent_0 = UnetGruWithStates2D(shape=image_size, num_filters_base=num_filters_base, num_levels=num_levels, data_format=data_format, normalization=normalization, is_training=is_training, name='unet_0', padding=padding)
    unet_recurrent_cell_0 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_0, shape=image_size, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, is_training=is_training, name='unet_0_cell', padding=padding)
    lstm_input_states = unet_recurrent_cell_0.zero_state(1, tf.float32)
    embeddings, states = unet_recurrent_cell_0(input, lstm_input_states)
    #embeddings = tf.transpose(embeddings_transposed, [0, 2, 1, 3, 4])
    embeddings_normalized = tf.nn.l2_normalize(embeddings, dim=1)
    input_lstm_1 = tf.concat([input, embeddings_normalized], name='embedding_input_concat', axis=1)
    unet_recurrent_1 = UnetGruWithStates2D(shape=image_size, num_filters_base=num_filters_base_2, num_levels=num_levels, data_format=data_format, normalization=normalization, is_training=is_training, name='unet_1', padding=padding)
    unet_recurrent_cell_1 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_1, shape=image_size, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, is_training=is_training, name='unet_1_cell', padding=padding)
    lstm_input_states_1 = unet_recurrent_cell_1.zero_state(1, tf.float32)
    embeddings_2, states_2 = unet_recurrent_cell_1(input_lstm_1, lstm_input_states_1)
    lstm_input_states = lstm_input_states + lstm_input_states_1
    lstm_output_states = states + states_2
    return embeddings, embeddings_2, lstm_input_states, lstm_output_states

