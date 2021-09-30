import tensorflow as tf
from tensorflow_train_v2.layers.layers import UpSampling3DLinear, UpSampling3DCubic
from tensorflow.keras.layers import AveragePooling3D


def salt_pepper_3D(image, rate, scales=[1], data_format='channels_first', seed=None):
    image = tf.convert_to_tensor(image, name="image")
    rate = tf.convert_to_tensor(rate, dtype=image.dtype, name="rate")
    ret = image
    for scale in scales:
        pooling_layer = AveragePooling3D([scale] * 3, data_format=data_format)
        upsampling_layer = UpSampling3DLinear([scale] * 3, data_format=data_format)
        dummy = tf.zeros_like(image)
        cur_shape = tf.shape(pooling_layer(dummy))
        random_tensor = tf.random.uniform(cur_shape, seed=seed, dtype=image.dtype)
        zeros_mask = tf.cast(random_tensor < rate / 2, image.dtype)
        ones_mask = tf.cast(random_tensor >= 1 - rate / 2, image.dtype)
        zeros_mask = upsampling_layer(zeros_mask)
        ones_mask = upsampling_layer(ones_mask)
        ret = (1 - (1 - ret) * (1 - ones_mask)) * (1 - zeros_mask)
    return ret


