
import tensorflow as tf

he_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
selu_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
zeros_initializer = tf.zeros_initializer
