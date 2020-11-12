
import tensorflow.compat.v1 as tf
from tensorflow_train.utils.tensorflow_util import reduce_mean_masked, reduce_sum_masked, masked_bit, save_divide, save_reduce_mean, most_significant_bit, reduce_median_masked


def cosine_embedding_single_instance_loss(embeddings, target_instances_mask, other_instances_mask, normalized_embeddings=True, term_1_2_normalization='individual', term_0_squared=False, term_1_squared=False, return_h_norm=False,  use_first_frame_for_mean=False, is_background=False, data_format='channels_first'):
    if data_format == 'channels_first':
        embedding_axis = 0
        frame_axis = 1
        if embeddings.shape.ndims == 3:
            image_axes = [1, 2]
        elif embeddings.shape.ndims == 4:
            image_axes = [1, 2, 3]
    else:
        embedding_axis = embeddings.shape.ndims - 1
        frame_axis = 0
        if embeddings.shape.ndims == 3:
            image_axes = [0, 1]
        elif embeddings.shape.ndims == 4:
            image_axes = [0, 1, 2]
    # expand axis, such that embeddings and instances are in different dimensions
    # create target and other instances pixel masks
    # calculate mean embedding for target pixels
    if use_first_frame_for_mean:
        # FIXME: does not support videos/volumes
        slices = [slice(None)] * 4
        slices.insert(frame_axis, slice(0, 1))
        h = reduce_sum_masked(embeddings[slices], target_instances_mask[slices], axis=image_axes, keepdims=True)
    else:
        if not is_background:
            h = reduce_mean_masked(embeddings, target_instances_mask, axis=image_axes, keepdims=True)
        else:
            if embeddings.shape.ndims == 3:
                if term_1_squared:
                    h = tf.concat([tf.ones((1, 1, 1), dtype=embeddings.dtype), tf.zeros((embeddings.shape[embedding_axis] - 1, 1, 1), dtype=embeddings.dtype)], axis=0)
                else:
                    h = tf.ones((embeddings.shape[embedding_axis], 1, 1)) * (-1)
            elif embeddings.shape.ndims == 4:
                if term_1_squared:
                    h = tf.concat([tf.ones((1, 1, 1, 1), dtype=embeddings.dtype), tf.zeros((embeddings.shape[embedding_axis] - 1, 1, 1, 1), dtype=embeddings.dtype)], axis=0)
                else:
                    h = tf.ones((embeddings.shape[embedding_axis], 1, 1, 1)) * (-1)
    h_norm = tf.nn.l2_normalize(h, dim=embedding_axis) #, epsilon=1e-4)
    # l2_normalize embeddings -> needed for cos_simliarity
    if normalized_embeddings is None:
        embeddings = tf.nn.l2_normalize(embeddings, dim=embedding_axis)
    else:
        embeddings = normalized_embeddings
    # calculate cos_similarity with target mean embedding and all embeddings
    cos_similarity = tf.reduce_sum(h_norm * embeddings, axis=embedding_axis, keepdims=True)
    # term_0: target mean embedding and target pixel embeddings should be as similar as possible
    if term_0_squared:
        term_0 = 1 - (cos_similarity ** 2)
    else:
        term_0 = 1 - cos_similarity
    if term_1_squared:
        # term_1: target mean embedding and other pixel embeddings should be orthogonal (== 0)
        term_1 = cos_similarity ** 2
    else:
        # term_1: target mean embedding and other pixel embeddings should be far apart (>= 0)
        term_1 = tf.nn.relu(cos_similarity)

    if term_1_2_normalization == 'individual':
        term_0 = tf.expand_dims(reduce_mean_masked(term_0, target_instances_mask), axis=0)
        term_1 = tf.expand_dims(reduce_mean_masked(term_1, other_instances_mask), axis=0)
        return term_0, term_1
    elif term_1_2_normalization == 'none' or term_1_2_normalization == 'combined':
        term_0 = tf.boolean_mask(term_0, target_instances_mask)
        term_1 = tf.boolean_mask(term_1, other_instances_mask)
        if not return_h_norm:
            return term_0, term_1
        else:
            return term_0, term_1, tf.squeeze(h_norm)
    else:
        assert 'invalid normalization mode'


def cosine_embedding_all_instances_loss(embeddings, instances, normalized_embeddings, term_1_2_normalization, term_0_squared, term_1_squared, return_h_norm, is_background, instances_used, data_format, parallel_iterations):
    instance_axis = 0 if data_format == 'channels_first' else instances.shape.ndims - 1
    i = tf.constant(0, tf.shape(instances).dtype)
    num_instances = tf.shape(instances)[instance_axis]
    ta_term_0 = tf.TensorArray(embeddings.dtype, size=num_instances, infer_shape=False)
    ta_term_1 = tf.TensorArray(embeddings.dtype, size=num_instances, infer_shape=False)
    ta_term_2 = tf.TensorArray(embeddings.dtype, size=num_instances, infer_shape=False)

    cond = lambda i, *args: i < num_instances

    def per_instance_loss(i, current_ta_term_0, current_ta_term_1, current_ta_term_2):
        # TODO: simplify condition, use instance_axis somehow
        if data_format == 'channels_first':
            target_instances_mask = tf.expand_dims(tf.equal(instances[i, ...], 1), axis=instance_axis)
            other_instances_mask = tf.expand_dims(tf.equal(instances[i, ...], 2), axis=instance_axis)
        else:
            target_instances_mask = tf.expand_dims(tf.equal(instances[..., i], 1), axis=instance_axis)
            other_instances_mask = tf.expand_dims(tf.equal(instances[..., i], 2), axis=instance_axis)

        current_terms = tf.cond(tf.greater(tf.count_nonzero(target_instances_mask), 0),
                                lambda: cosine_embedding_single_instance_loss(embeddings, target_instances_mask, other_instances_mask, normalized_embeddings, term_1_2_normalization=term_1_2_normalization, term_0_squared=term_0_squared, term_1_squared=term_1_squared, return_h_norm=return_h_norm, is_background=is_background, data_format=data_format),
                                lambda: (tf.zeros([0], dtype=embeddings.dtype), tf.zeros([0], dtype=embeddings.dtype)) if not return_h_norm else (tf.zeros([0], dtype=embeddings.dtype), tf.zeros([0], dtype=embeddings.dtype), tf.zeros([embeddings.shape[0]], dtype=embeddings.dtype)))
        current_ta_term_0 = current_ta_term_0.write(i, current_terms[0])
        current_ta_term_1 = current_ta_term_1.write(i, current_terms[1])
        if return_h_norm:
            current_ta_term_2 = current_ta_term_2.write(i, current_terms[2])
        return i + 1, current_ta_term_0, current_ta_term_1, current_ta_term_2

    _, ta_term_0_final, ta_term_1_final, ta_term_2_final = tf.while_loop(cond, per_instance_loss, (i, ta_term_0, ta_term_1, ta_term_2), parallel_iterations=parallel_iterations, swap_memory=True)

    term_0 = tf.cond(tf.equal(num_instances, tf.constant(0, num_instances.dtype)), lambda: tf.zeros([0], dtype=embeddings.dtype), lambda: ta_term_0_final.concat())
    term_1 = tf.cond(tf.equal(num_instances, tf.constant(0, num_instances.dtype)), lambda: tf.zeros([0], dtype=embeddings.dtype), lambda: ta_term_1_final.concat())
    if return_h_norm:
        term_2 = tf.cond(tf.equal(num_instances, tf.constant(0, num_instances.dtype)), lambda: tf.zeros([0], dtype=embeddings.dtype), lambda: ta_term_2_final.stack())
        return term_0, term_1, term_2
    else:
        return term_0, term_1


def cosine_embedding_all_instances_bitwise_loss(embeddings, instances_bitwise, normalized_embeddings, term_1_2_normalization, term_0_squared, term_1_squared, return_h_norm, is_background, instances_used, data_format, parallel_iterations):
    assert instances_bitwise.dtype in [tf.int8, tf.int16, tf.int32, tf.int64], 'unsupported data type, must be int*'

    # calculate num_instances as the bit index of the highest bit / 2
    num_instances = tf.cast(most_significant_bit(tf.reduce_max(instances_bitwise)) / 2, tf.int32)

    i = tf.constant(0, tf.int32)
    ta_term_0 = tf.TensorArray(embeddings.dtype, size=num_instances, infer_shape=False)
    ta_term_1 = tf.TensorArray(embeddings.dtype, size=num_instances, infer_shape=False)
    ta_term_2 = tf.TensorArray(embeddings.dtype, size=num_instances, infer_shape=False)

    cond = lambda i, *args: i < num_instances

    def per_instance_loss(i, current_ta_term_0, current_ta_term_1, current_ta_term_2):
        target_instances_mask = masked_bit(instances_bitwise, i * 2)
        other_instances_mask = masked_bit(instances_bitwise, i * 2 + 1)
        current_terms = tf.cond(tf.greater(tf.count_nonzero(target_instances_mask), 0),
                                lambda: cosine_embedding_single_instance_loss(embeddings, target_instances_mask, other_instances_mask, normalized_embeddings, term_1_2_normalization=term_1_2_normalization, term_0_squared=term_0_squared, term_1_squared=term_1_squared, return_h_norm=return_h_norm, is_background=is_background, data_format=data_format),
                                lambda: (tf.zeros([0], dtype=embeddings.dtype), tf.zeros([0], dtype=embeddings.dtype)) if not return_h_norm else (tf.zeros([0], dtype=embeddings.dtype), tf.zeros([0], dtype=embeddings.dtype), tf.zeros([embeddings.shape[0]], dtype=embeddings.dtype)))
        current_ta_term_0 = current_ta_term_0.write(i, current_terms[0])
        current_ta_term_1 = current_ta_term_1.write(i, current_terms[1])
        if return_h_norm:
            current_ta_term_2 = current_ta_term_2.write(i, current_terms[2])
        return i + 1, current_ta_term_0, current_ta_term_1, current_ta_term_2

    _, ta_term_0_final, ta_term_1_final, ta_term_2_final = tf.while_loop(cond, per_instance_loss, (i, ta_term_0, ta_term_1, ta_term_2), parallel_iterations=parallel_iterations, swap_memory=True)

    term_0 = tf.cond(tf.equal(num_instances, tf.constant(0, num_instances.dtype)), lambda: tf.zeros([0], dtype=embeddings.dtype), lambda: ta_term_0_final.concat())
    term_1 = tf.cond(tf.equal(num_instances, tf.constant(0, num_instances.dtype)), lambda: tf.zeros([0], dtype=embeddings.dtype), lambda: ta_term_1_final.concat())
    if return_h_norm:
        term_2 = tf.cond(tf.equal(num_instances, tf.constant(0, num_instances.dtype)), lambda: tf.zeros([0, embeddings.shape[0]], dtype=embeddings.dtype), lambda: ta_term_2_final.stack())
        return term_0, term_1, term_2
    else:
        return term_0, term_1


def cosine_embedding_per_instance_loss(embeddings, instances, normalized_embeddings=True, bitwise_instances=False, term_1_2_normalization='individual', l=1.0, term_0_squared=False, term_1_squared=False, return_h_norm=False, is_background=False, instances_used=None, data_format='channels_first', parallel_iterations=8):
    if not normalized_embeddings:
        embeddings_l2_normalized = tf.nn.l2_normalize(embeddings, dim=0)
    else:
        embeddings_l2_normalized = embeddings
    with tf.name_scope('cosine_embedding_loss'):
        if not bitwise_instances:
            terms = cosine_embedding_all_instances_loss(embeddings, instances, embeddings_l2_normalized, term_1_2_normalization, term_0_squared, term_1_squared, return_h_norm, is_background, instances_used, data_format, parallel_iterations)
        else:
            terms = cosine_embedding_all_instances_bitwise_loss(embeddings, instances, embeddings_l2_normalized, term_1_2_normalization, term_0_squared, term_1_squared, return_h_norm, is_background, instances_used, data_format, parallel_iterations)

        if term_1_2_normalization == 'individual':
            return save_reduce_mean(terms[0] + l * terms[1])
        elif term_1_2_normalization == 'combined':
            term_0 = save_reduce_mean(terms[0])
            term_1 = save_reduce_mean(terms[1])
            return save_reduce_mean(term_0 + l * term_1)
        elif term_1_2_normalization == 'none':
            return terms
        else:
            assert 'invalid normalization mode'


def cosine_embedding_per_instance_batch_loss(embeddings, instances, normalized_embeddings=True, bitwise_instances=False, term_1_2_normalization='individual', l=1.0, l2=0.0, term_0_squared=False, term_1_squared=False, is_background=False, instances_used=None, data_format='channels_first', parallel_iterations=8):
    loss_list_term_0 = []
    loss_list_term_1 = []
    loss_list_term_2 = []
    return_h_norm = l2 > 0.0
    for i in range(embeddings.get_shape().as_list()[0]):
        if instances_used is not None:
            current_instances_used = instances_used[i, ...]
        else:
            current_instances_used = None
        current_loss = cosine_embedding_per_instance_loss(embeddings[i, ...], instances[i, ...],
                                                          normalized_embeddings, bitwise_instances, term_1_2_normalization, l, term_0_squared, term_1_squared, return_h_norm, is_background, current_instances_used, data_format, parallel_iterations)
        if isinstance(current_loss, tuple):
            loss_list_term_0.append(current_loss[0])
            loss_list_term_1.append(current_loss[1])
            if len(current_loss) > 2:
                loss_list_term_2.append(current_loss[2])
        else:
            loss_list_term_0.append(current_loss)
    total_loss = save_reduce_mean(tf.concat(loss_list_term_0, axis=0))
    if len(loss_list_term_1) > 0:
        total_loss += l * save_reduce_mean(tf.concat(loss_list_term_1, axis=0))
    if len(loss_list_term_2) > 0:
        total_loss += l2 * save_reduce_mean(tf.linalg.norm(tf.concat(loss_list_term_2, axis=0), 1, axis=1))
    return total_loss

