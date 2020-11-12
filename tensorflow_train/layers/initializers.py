
import tensorflow.compat.v1 as tf

he_initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal')
selu_initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_in', distribution='truncated_normal')
zeros_initializer = tf.zeros_initializer
