
import tensorflow as tf

from tensorflow_train_v2.layers.upsample import upsample_linear, upsample_cubic, upsample_lanczos
from tensorflow_train_v2.utils.data_format import get_batch_channel_image_size_from_shape_tuple, create_tensor_shape_tuple


class Sequential(tf.keras.layers.Layer):
    """
    A keras layer that applies a list of encapsulated layers sequentially.
    """
    def __init__(self, layers, *args, **kwargs):
        """
        Initializer.
        :param layers: The list of layers to apply sequentially.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(Sequential, self).__init__(*args, **kwargs)
        self.layers = layers

    def call(self, inputs, **kwargs):
        """
        Call the internal layers sequentially for the given inputs.
        :param inputs: The layer inputs.
        :param kwargs: **kwargs
        :return: The output of the last layer.
        """
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class Residual(tf.keras.layers.Layer):
    """
    A keras layer that applies a list of encapsulated layers as a residual unit.
    """
    def __init__(self, layers_before_residual, layers_after_residual, *args, **kwargs):
        """
        Initializer.
        :param layers: The list of layers to apply sequentially.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(Residual, self).__init__(*args, **kwargs)
        self.layers_before_residual = layers_before_residual
        self.layers_after_residual = layers_after_residual
        self.layers = layers_before_residual + layers_after_residual

    def call(self, inputs, **kwargs):
        """
        Call the internal layers sequentially for the given inputs.
        :param inputs: The layer inputs.
        :param kwargs: **kwargs
        :return: The output of the last layer.
        """
        node = inputs
        for layer in self.layers_before_residual:
            node = layer(node)
        node += inputs
        for layer in self.layers_before_residual:
            node = layer(node)
        return node


class ConcatChannels(tf.keras.layers.Layer):
    """
    Concat the channel dimension.
    """
    def __init__(self, data_format=None, *args, **kwargs):
        """
        Initializer
        :param data_format: The data_format.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(ConcatChannels, self).__init__(*args, **kwargs)
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        """
        Computes the output_shape.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        if input_shape is None:
            return None
        all_channel_sizes = [get_batch_channel_image_size_from_shape_tuple(s, self.data_format)[1] if s is not None else None for s in input_shape]
        if None in all_channel_sizes:
            return None
        return sum(all_channel_sizes)

    def call(self, inputs, **kwargs):
        """
        Concatenate the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The concatenated outputs.
        """
        axis = 1 if self.data_format == 'channels_first' else -1
        return tf.concat(inputs, axis=axis)


class UpSamplingBase(tf.keras.layers.Layer):
    """
    UpSampling base layer.
    """
    def __init__(self, dim, size, data_format, *args, **kwargs):
        """
        Initializer.
        :param size: The scaling factors. Must be integer.
        :param data_format: The data_format.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(UpSamplingBase, self).__init__(*args, **kwargs)
        self.dim = dim
        self.size = size
        if self.dim != len(self.size):
            raise ValueError(f'dim and size parameter do not agree, dim: {dim}, size: {size}')
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        """
        Computes the output_shape.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        if input_shape is None:
            return None
        if not isinstance(input_shape, tuple):
            # when input_shape is a list, multiple inputs were defined
            raise ValueError('UpSampling layers only allow a single input')
        if len(input_shape) != self.dim + 2:
            raise ValueError(f'Dimension of input tensor is invalid, len(input_shape): {len(input_shape)}, expected: {self.dim + 2}')
        batch_size, channel_size, image_size = get_batch_channel_image_size_from_shape_tuple(input_shape, self.data_format)
        new_image_size = tuple([size * scale if size is not None else None for size, scale in zip(image_size, self.size)])
        return create_tensor_shape_tuple(batch_size, channel_size, new_image_size, self.data_format)


class UpSampling2DLinear(UpSamplingBase):
    """
    UpSampling 3D with linear interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling2DLinear, self).__init__(2, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_linear(inputs, self.size, self.data_format)


class UpSampling3DLinear(UpSamplingBase):
    """
    UpSampling 3D with linear interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling3DLinear, self).__init__(3, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_linear(inputs, self.size, self.data_format)


class UpSampling2DCubic(UpSamplingBase):
    """
    UpSampling 2D with cubic interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling2DCubic, self).__init__(2, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_cubic(inputs, self.size, self.data_format)


class UpSampling3DCubic(UpSamplingBase):
    """
    UpSampling 3D with cubic interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling3DCubic, self).__init__(3, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_cubic(inputs, self.size, self.data_format)


class UpSampling2DLanczos(UpSamplingBase):
    """
    UpSampling 2D with lanczos interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling2DLanczos, self).__init__(2, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_lanczos(inputs, self.size, self.data_format)


class UpSampling3DLanczos(UpSamplingBase):
    """
    UpSampling 3D with lanczos interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling3DLanczos, self).__init__(3, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_lanczos(inputs, self.size, self.data_format)
