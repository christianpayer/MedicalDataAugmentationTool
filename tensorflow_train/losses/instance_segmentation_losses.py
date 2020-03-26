
import tensorflow as tf
from tensorflow_train.utils.data_format import get_image_axes
from tensorflow_train.utils.tensorflow_util import reduce_mean_masked, reduce_sum_masked


def cosine_embedding_single_instance_loss(embeddings, target_instances_mask, other_instances_mask, embeddings_norm=None, normalize=False, l=1.0, term_1_squared=False, term_2_factor=0, use_first_frame_for_mean=False, data_format='channels_first'):
    image_axes = get_image_axes(embeddings, data_format)
    image_axes = [i for i in image_axes]
    if data_format == 'channels_first':
        embedding_axis = 1
        frame_axis = 2
    else:
        embedding_axis = len(embeddings.shape) - 1
        frame_axis = 1
    # expand axis, such that embeddings and instances are in different dimensions
    # create target and other instances pixel masks
    #target_instances_mask = tf.equal(instances, 1)
    #other_instances_mask = tf.equal(instances, 2)
    # calculate mean embedding for target pixels
    if use_first_frame_for_mean:
        slices = [slice(None)] * 4
        slices.insert(frame_axis, slice(0, 1))
        h = reduce_mean_masked(embeddings[slices], target_instances_mask[slices], axis=image_axes, keepdims=True)
    else:
        h = reduce_mean_masked(embeddings, target_instances_mask, axis=image_axes, keepdims=True)
    if embeddings_norm is None:
        embeddings_norm = tf.nn.l2_normalize(embeddings, dim=embedding_axis)
    # l2_normalize embeddings -> needed for cos_simliarity
    h_norm = tf.nn.l2_normalize(h, dim=embedding_axis)
    # calculate cos_similarity with target mean embedding and all embeddings
    cos_similarity = tf.reduce_sum(h_norm * embeddings_norm, axis=embedding_axis, keepdims=True)
    # term_0: target mean embedding and target pixel embeddings should be as similar as possible
    term_0 = 1 - cos_similarity
    if term_1_squared:
        # term_1: target mean embedding and other pixel embeddings should be orthogonal (== 0)
        term_1 = cos_similarity ** 2
    else:
        # term_1: target mean embedding and other pixel embeddings should be far apart (>= 0)
        term_1 = tf.nn.relu(cos_similarity)

    # either reduce_mean or reduce_sum on target and other pixel masks
    if normalize:
        term_0 = reduce_mean_masked(term_0, target_instances_mask)
        term_1 = reduce_mean_masked(term_1, other_instances_mask)
    else:
        term_0 = reduce_sum_masked(term_0, target_instances_mask)
        term_1 = reduce_sum_masked(term_1, other_instances_mask)

    term_2 = 0
    if term_2_factor > 0:
        instance_mask = tf.reduce_any(target_instances_mask, axis=image_axes, keepdims=True)
        term_2 = tf.norm(h_norm, ord=1, axis=embedding_axis, keepdims=True)
        term_2 = reduce_mean_masked(term_2, instance_mask) * term_2_factor

    loss = term_0 + l * term_1 + term_2
    return loss


def cosine_embedding_per_instance_loss(embeddings, instances, normalize=False, l=1.0, term_1_squared=False, term_2_factor=0, data_format='channels_first', parallel_iterations=4):
    image_axes = get_image_axes(embeddings, data_format)
    #image_axes = [i + 1 for i in image_axes]
    if data_format == 'channels_first':
        embedding_axis = 1
    else:
        embedding_axis = len(embeddings.shape) - 1
    if len(embeddings.shape) == 5:
        instances_transposed = tf.expand_dims(tf.transpose(instances, [1, 0, 2, 3, 4]), axis=2)
    else:
        instances_transposed = tf.expand_dims(tf.transpose(instances, [1, 0, 2, 3]), axis=2)
    print(instances_transposed.shape)
    embeddings_norm = tf.nn.l2_normalize(embeddings, dim=embedding_axis)
    per_instance_loss = lambda i: cosine_embedding_single_instance_loss(embeddings, tf.equal(i, 1), tf.equal(i, 2), embeddings_norm=embeddings_norm, normalize=normalize, l=l, term_1_squared=term_1_squared, data_format=data_format, term_2_factor=term_2_factor)
    loss_list = tf.map_fn(per_instance_loss, instances_transposed, swap_memory=True, dtype=tf.float32, parallel_iterations=parallel_iterations)
    return tf.reduce_mean(loss_list)
