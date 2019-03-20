
import tensorflow as tf

def create_reset_metric(metric, variable_scope, **metric_args):
    with tf.variable_scope(variable_scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
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


def create_placeholders(name_shape_dict, shape_prefix=None, shape_postfix=None):
    if shape_prefix is None:
        shape_prefix = []
    if shape_postfix is None:
        shape_postfix = []
    return dict([(name, tf.placeholder(tf.float32, shape_prefix + shape + shape_postfix, name='placeholder_' + name))
                 for (name, shape) in name_shape_dict.items()])


def reduce_sum_weighted(input, weights, axis=None, keep_dims=False):
    input_masked = input * weights
    return tf.reduce_sum(input_masked, axis=axis, keep_dims=keep_dims)


def reduce_mean_weighted(input, weights, axis=None, keep_dims=False):
    input_masked = input * weights
    sum = tf.reduce_sum(input_masked, axis=axis, keep_dims=keep_dims)
    # TODO: change to save_divide, when implemented in tensorflow
    # bugfix for nan propagation:
    # set num_elements to 1, when they are actually zero. this will not influence the output value, as sum will be 0 in this case as well
    num_elements = tf.reduce_sum(weights, axis=axis, keep_dims=keep_dims)
    num_elements = tf.where(num_elements > 0, num_elements, tf.ones_like(num_elements))
    return sum / num_elements


def reduce_sum_masked(input, mask, axis=None, keep_dims=False):
    assert mask.dtype == tf.bool, 'mask must be bool'
    # convert mask to float and use it as weights
    weights = tf.cast(mask, dtype=input.dtype)
    return reduce_sum_weighted(input, weights, axis, keep_dims)
    #bad_data, good_data = tf.dynamic_partition(input, tf.cast(mask, tf.int32), 2)
    #return tf.reduce_sum(bad_data, axis=axis, keep_dims=keep_dims)


def reduce_mean_masked(input, mask, axis=None, keep_dims=False):
    assert mask.dtype == tf.bool, 'mask must be bool'
    # convert mask to float and use it as weights
    weights = tf.cast(mask, dtype=input.dtype)
    return reduce_mean_weighted(input, weights, axis, keep_dims)


def reduce_mean_support_empty(input):
    return tf.cond(tf.size(input) > 0, lambda: tf.reduce_mean(input), lambda: tf.constant(0, dtype=input.dtype))