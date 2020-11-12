
from collections import OrderedDict
import tensorflow.compat.v1 as tf


def create_reset_metric(metric, variable_scope, **metric_args):
    """
    Creates tensors of a metric (e.g., running mean), its update, and reset operation.
    :param metric: The metric and its tensors to create.
    :param variable_scope: The variable scope which is needed to create the reset operation.
    :param metric_args: The args used for generating the metric.
    :return: Tensors of the metric, its update, and reset operation.
    """
    with tf.variable_scope(variable_scope):
        metric_op, update_op = metric(**metric_args)
        vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, variable_scope)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def print_progress_bar(iteration, total, prefix='Testing ', suffix=' complete', decimals=1, length=50, fill='X'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def create_placeholder(name, shape, shape_prefix=None, shape_postfix=None, data_type=None):
    """
    Creates a placeholder with name "'placeholder_' + name", shape (shape_prefix + shape + shape_postfix) and type data_type.
    :param name: The name of the placeholder will be 'placeholder_' + name.
    :param shape: The shape of the placeholder.
    :param shape_prefix: The shape prefix. Default is [].
    :param shape_postfix: The shape postfix. Default is [].
    :param data_type: The data type of the placeholder. Default dtype if not given is tf.float32.
    :return: The tf.placeholder.
    """
    shape_prefix = shape_prefix or []
    shape_postfix = shape_postfix or []
    data_type = data_type or tf.float32
    return tf.placeholder(data_type, shape_prefix + shape + shape_postfix, name='placeholder_' + name)


def create_placeholders(name_shape_dict, shape_prefix=None, shape_postfix=None, data_types=None):
    """
    Creates placeholders and returns them as an OrderedDict in the same order as name_shape_dict.
    :param name_shape_dict: Dict or OrderedDict with name as key and shape as value.
    :param shape_prefix: The shape prefix used for all shapes. Default is [].
    :param shape_postfix: The shape postfix used for all shapes. Default is [].
    :param data_types: The data types of all placeholders as a dict. Default dtype if not given is tf.float32.
    :return: The tuple of all placeholders.
    """
    shape_prefix = shape_prefix or []
    shape_postfix = shape_postfix or []
    data_types = data_types or {}
    return OrderedDict([(name, create_placeholder(name, shape, shape_prefix, shape_postfix, data_types.get(name, None)))
                        for (name, shape) in name_shape_dict.items()])


def create_placeholders_tuple(ordered_name_shape_dict, shape_prefix=None, shape_postfix=None, data_types=None):
    """
    Creates placeholders and returns them as a tuple in the same order as ordered_name_shape_dict.
    :param ordered_name_shape_dict: OrderedDict with name as key and shape as value.
    :param shape_prefix: The shape prefix used for all shapes. Default is [].
    :param shape_postfix: The shape postfix used for all shapes. Default is [].
    :param data_types: The data types of all placeholders as a dict. Default dtype if not given is tf.float32.
    :return: The tuple of all placeholders.
    """
    placeholders = create_placeholders(ordered_name_shape_dict, shape_prefix, shape_postfix, data_types).values()
    if len(placeholders) == 1:
        return tuple(placeholders)[0]
    else:
        return tuple(placeholders)


def save_divide(x, y):
    """
    Divides x by y. If y == 0, return x instead.
    :param x: Tensor.
    :param y: Tensor.
    :return: x / y
    """
    return x / tf.where(y > 0, y, tf.ones_like(y))


def save_reduce_mean(x, axis=None, keepdims=False):
    return save_divide(tf.reduce_sum(x, axis=axis, keepdims=keepdims), tf.cast(tf.count_nonzero(x, axis=axis, keepdims=keepdims), x.dtype))


def reduce_sum_weighted(input, weights, axis=None, keepdims=False):
    input_masked = input * weights
    return tf.reduce_sum(input_masked, axis=axis, keepdims=keepdims)


def reduce_mean_weighted(input, weights, axis=None, keepdims=False):
    input_masked = input * weights
    sum = tf.reduce_sum(input_masked, axis=axis, keepdims=keepdims)
    # TODO: change to save_divide, when implemented in tensorflow
    # bugfix for nan propagation:
    # set num_elements to 1, when they are actually zero. this will not influence the output value, as sum will be 0 in this case as well
    num_elements = tf.reduce_sum(weights, axis=axis, keepdims=keepdims)
    return save_divide(sum, num_elements)


def reduce_sum_masked(input, mask, axis=None, keepdims=False):
    assert mask.dtype == tf.bool, 'mask must be bool'
    # convert mask to float and use it as weights
    weights = tf.cast(mask, dtype=input.dtype)
    return reduce_sum_weighted(input, weights, axis, keepdims)
    #bad_data, good_data = tf.dynamic_partition(input, tf.cast(mask, tf.int32), 2)
    #return tf.reduce_sum(bad_data, axis=axis, keep_dims=keep_dims)


def reduce_mean_masked(input, mask, axis=None, keepdims=False):
    assert mask.dtype == tf.bool, 'mask must be bool'
    # convert mask to float and use it as weights
    weights = tf.cast(mask, dtype=input.dtype)
    return reduce_mean_weighted(input, weights, axis, keepdims)


def reduce_median(tensor, axis=None, keepdims=False):
    return tf.contrib.distributions.percentile(tensor, 50., axis=axis, keep_dims=keepdims)


def reduce_median_masked(tensor, mask, axis=None, keepdims=False):
    tensor_masked = tf.boolean_mask(tensor, mask)
    #print('shapes', tensor.shape, mask.shape, tensor_masked.shape)
    return tf.contrib.distributions.percentile(tensor_masked, 50., axis=axis, keep_dims=keepdims)


def reduce_mean_support_empty(input, keepdims=False):
    return tf.cond(tf.size(input) > 0, lambda: tf.reduce_mean(input, keepdims=keepdims), lambda: tf.zeros_like(input))


# def bit_tensor_list(input):
#     assert input.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64], 'unsupported data type, must be uint*'
#     num_bits = 0
#     if input.dtype == tf.int8:
#         num_bits = 8
#     elif input.dtype == tf.int16:
#         num_bits = 16
#     elif input.dtype == tf.uint32:
#         num_bits = 32
#     elif input.dtype == tf.uint64:
#         num_bits = 64
#     bit_tensors = []
#     for i in range(num_bits):
#         current_bit = 1 << i
#         current_bit_tensor = tf.bitwise.bitwise_and(input, current_bit) == 1
#         bit_tensors.append(current_bit_tensor)
#     print(bit_tensors)
#     return bit_tensors


def masked_bit(input, bit_index):
    """
    Returns a boolean tensor, where values are true, on which the bit on bit_index is True.
    :param input: The input tensor to check.
    :param bit_index: The bit index which will be compared with bitwise and. (LSB 0 order)
    :return: The tensor.
    """
    assert input.dtype in [tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32, tf.uint64], 'unsupported data type, must be *int*'
    current_bit = tf.bitwise.left_shift(tf.constant(1, dtype=input.dtype), tf.cast(bit_index, dtype=input.dtype))
    return tf.greater(tf.bitwise.bitwise_and(input, current_bit), 0)


def most_significant_bit(number):
    bitpos = tf.constant(0, number.dtype)
    cond = lambda current_number, _: current_number > 0
    shift_and_increment = lambda current_number, current_bitpos: (tf.bitwise.right_shift(current_number, tf.constant(1, current_number.dtype)), current_bitpos + 1)
    _, final_bitpos = tf.while_loop(cond, shift_and_increment, (number, bitpos))
    return final_bitpos


def get_reg_loss(reg_constant, collect_kernel_variables=False):
    """
    Returns the regularization loss for the regularized variables, multiplied with reg_constant.
    :param reg_constant: The multiplication factor.
    :param collect_kernel_variables: If true, uses all variables that contain the string 'kernel', otherwise uses tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES).
    :return: The regularization loss.
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if reg_constant > 0:
            if collect_kernel_variables:
                reg_losses = []
                for tf_var in tf.trainable_variables():
                    if 'kernel' in tf_var.name:
                        reg_losses.append(tf.nn.l2_loss(tf_var))
            else:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_reg = reg_constant * tf.add_n(reg_losses)
        else:
            loss_reg = 0
    return loss_reg


def masked_apply(tensor, op, mask, set_outside_zero=True):
    """
    Apply the function op to tensor only at locations indicated by mask. If set_outside_zero == True, set the
    locations outside the mask to zero, otherwise keep original value of tensor.
    :param tensor: The tensor on which op is applied.
    :param op: The operation.
    :param mask: The boolean mask.
    :param set_outside_zero: If True, set the locations outside the mask to zero, otherwise keep original values of tensor.
    :return: Tensor with applied function.
    """
    chosen = tf.boolean_mask(tensor, mask)
    applied = op(chosen)
    idx = tf.to_int32(tf.where(mask))
    result = tf.scatter_nd(idx, applied, tf.shape(tensor))
    if not set_outside_zero:
        result = tf.where(mask, result, tensor)
    return result
