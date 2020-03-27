
import tensorflow as tf
from tensorflow_train.networks.unet_lstm_dynamic import UnetGruWithStates2D, UnetRecurrentCell2D


def network_single_frame_with_lstm_states(input, num_outputs_embedding, data_format='channels_first'):
    if data_format == 'channels_first':
        image_size = input.get_shape().as_list()[2:4]
        channel_axis = 1
    else:
        image_size = input.get_shape().as_list()[1:3]
        channel_axis = 3
    normalization = None
    is_training = False
    num_levels = 7
    num_filters_base = 64
    num_filters_base_2 = 64

    unet_recurrent_0 = UnetGruWithStates2D(shape=image_size, num_filters_base=num_filters_base, num_levels=num_levels, data_format=data_format, normalization=normalization, is_training=is_training, name='unet_0')
    unet_recurrent_cell_0 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_0, shape=image_size, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, is_training=is_training, name='unet_0_cell')
    lstm_input_states_0 = unet_recurrent_cell_0.zero_state(1, tf.float32)
    embeddings_0, states_0 = unet_recurrent_cell_0(input, lstm_input_states_0)
    embeddings_normalized_0 = tf.nn.l2_normalize(embeddings_0, dim=channel_axis)

    input_lstm_1 = tf.concat([input, embeddings_normalized_0], name='embedding_input_concat', axis=channel_axis)
    unet_recurrent_1 = UnetGruWithStates2D(shape=image_size, num_filters_base=num_filters_base_2, num_levels=num_levels, data_format=data_format, normalization=normalization, is_training=is_training, name='unet_1')
    unet_recurrent_cell_1 = UnetRecurrentCell2D(unet_recurrent=unet_recurrent_1, shape=image_size, num_outputs=num_outputs_embedding, kernel=[3, 3], data_format=data_format, is_training=is_training, name='unet_1_cell')
    lstm_input_states_1 = unet_recurrent_cell_1.zero_state(1, tf.float32)
    embeddings_1, states_1 = unet_recurrent_cell_1(input_lstm_1, lstm_input_states_1)

    lstm_input_states = lstm_input_states_0 + lstm_input_states_1
    lstm_output_states = states_0 + states_1
    return embeddings_0, embeddings_1, lstm_input_states, lstm_output_states

