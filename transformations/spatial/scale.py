import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class ScaleTransformBase(SpatialTransformBase):
    """
    Scale transformation base class.
    """
    @staticmethod
    def get_scale_transform(dim, scale):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param scale: List of scale factors for each dimension.
        :return: The sitk.AffineTransform().
        """
        if isinstance(scale, list) or isinstance(scale, tuple):
            assert len(scale) == dim, 'Length of scale must be equal to dim.'

        s = sitk.AffineTransform(dim)
        s.Scale(scale)

        return s


class Fixed(ScaleTransformBase):
    """
    A scale transformation with fixed scaling factors.
    """
    def __init__(self, dim, scale):
        """
        Initializer.
        :param dim: The dimension.
        :param scale: List of scaling factors for each dimension.
        """
        super(Fixed, self).__init__(dim)
        self.current_scale = scale

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        return self.get_scale_transform(self.dim, self.current_scale)


class Random(ScaleTransformBase):
    """
    A scale transformation with random scaling factors.
    """
    def __init__(self, dim, random_scale):
        """
        Initializer.
        :param dim: The dimension.
        :param random_scale: List of random scaling factors per dimension. A random scaling factor is calculated uniformly within [1.0 -random_scale[i], 1.0 + random_scale[i])]
        """
        super(ScaleTransformBase, self).__init__(dim)
        self.random_scale = random_scale

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        current_scale = [1.0 + float_uniform(-self.random_scale[i], self.random_scale[i])
                         for i in range(len(self.random_scale))]
        return self.get_scale_transform(self.dim, current_scale)


class RandomUniform(ScaleTransformBase):
    """
    A scale transformation with a random scaling factor, equal for each dimension.
    """
    def __init__(self, dim, random_scale):
        """
        Initializer.
        :param dim: The dimension.
        :param random_scale: Single value of random scaling factor used for every dimension. A random scaling factor is calculated uniformly within [1.0 -random_scale[i], 1.0 + random_scale[i])]
        """
        super(ScaleTransformBase, self).__init__(dim)
        self.random_scale = random_scale

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        current_scale = 1.0 + float_uniform(-self.random_scale, self.random_scale)
        return self.get_scale_transform(self.dim, current_scale)


class Fit(ScaleTransformBase):
    """
    A scale transformation that scales the input image such that it fits in the defined output image.
    This may change the aspect ratio of the image!
    """
    def __init__(self, dim, output_size, output_spacing=None):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output spacing in mm.
        """
        super(Fit, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        current_scale = [(input_size[i] * input_spacing[i]) / (self.output_size[i] * self.output_spacing[i]) for i in range(self.dim)]
        return self.get_scale_transform(self.dim, current_scale)


class FitFixedAr(ScaleTransformBase):
    """
    A scale transformation that scales the input image such that it fits in the defined output image
    without changing the aspect ratio of the image.
    """
    def __init__(self, dim, output_size, output_spacing=None):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output spacing in mm.
        """
        super(FitFixedAr, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        scale = max([(input_size[i] * input_spacing[i]) / (self.output_size[i] * self.output_spacing[i]) for i in range(self.dim)])
        current_scale = [scale] * self.dim
        return self.get_scale_transform(self.dim, current_scale)


class InputSpacingToUniformSpacing(ScaleTransformBase):
    """
    A scale transformation that scales the input image such that each pixel has a physical spacing of 1 mm.
    The calculated scaling factor is 1 / input_spacing[i] for each dimension.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        current_scale = [1 / input_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(self.dim, current_scale)


class UniformSpacingToInputSpacing(ScaleTransformBase):
    """
    A scale transformation that scales each pixel (with expected physical spacing of 1 mm) such that it has the spacing of the input image.
    The calculated scaling factor is input_spacing[i] for each dimension.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        _, input_spacing = self.get_image_size_spacing(**kwargs)
        current_scale = [input_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(self.dim, current_scale)


class OutputSpacingToUniformSpacing(ScaleTransformBase):
    """
    A scale transformation that scales the output image such that each pixel has a physical spacing of 1 mm.
    The calculated scaling factor is 1 / output_spacing[i] for each dimension.
    """
    def __init__(self, dim, output_spacing):
        """
        Initializer.
        :param dim: The dimension.
        :param output_spacing: The output spacing.
        """
        super(OutputSpacingToUniformSpacing, self).__init__(dim)
        self.output_spacing = output_spacing

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        current_scale = [1 / self.output_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(self.dim, current_scale)


class UniformSpacingToOutputSpacing(ScaleTransformBase):
    """
    A scale transformation that scales each pixel (with expected physical spacing of 1 mm) such that it has the spacing of the output image.
    The calculated scaling factor is output_spacing[i] for each dimension.
    """
    def __init__(self, dim, output_spacing):
        """
        Initializer.
        :param dim: The dimension.
        :param output_spacing: The output spacing.
        """
        super(UniformSpacingToOutputSpacing, self).__init__(dim)
        self.output_spacing = output_spacing

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        current_scale = [self.output_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(self.dim, current_scale)
