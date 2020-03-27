
import tensorflow as tf

he_initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
zeros_initializer = tf.zeros_initializer
