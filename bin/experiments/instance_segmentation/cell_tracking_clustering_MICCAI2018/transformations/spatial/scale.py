import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class ScaleTransformBase(SpatialTransformBase):
    def __init__(self, dim):
        super(ScaleTransformBase, self).__init__(dim)

    def get_flipped(self):
        return [False] * self.dim

    def get_scale_transform(self, scale):
        """
        Get the scale transform without executing it on an image.

        :param scale: float, or list of floats
            the scale factor for each dimension
        :return:
        """
        # use the affine transform
        s = sitk.AffineTransform(self.dim)
        s.Scale(scale)

        # return the scale transform
        return s

class Fixed(ScaleTransformBase):
    def __init__(self, dim, scale):
        super(Fixed, self).__init__(dim)
        self.current_scale = scale

    def get(self, **kwargs):
        return self.get_scale_transform(self.current_scale)

class Random(ScaleTransformBase):
    def __init__(self, dim, random_scale):
        super(ScaleTransformBase, self).__init__(dim)
        self.random_scale = random_scale

    def get(self, **kwargs):
        """
        Apply random scaling, in each dimension. The scale factor will be computed from a random uniform
        distribution 1+[-random_scale, random_scale].
        

        :param input_image: ITK image
        :param random_scale: float, list of floats
            ranges for uniform random scaling in each dimension
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        # scale by individual factor in each dimension
        if isinstance(self.random_scale, list) or isinstance(self.random_scale, tuple):
            current_scale = [1.0 + float_uniform(-self.random_scale[i], self.random_scale[i])
                             for i in range(len(self.random_scale))]
        else:
            current_scale = 1.0 + float_uniform(-self.random_scale, self.random_scale)
        return self.get_scale_transform(current_scale)

class Fit(ScaleTransformBase):
    def __init__(self, dim, output_size, output_spacing=None):
        super(Fit, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get(self, **kwargs):
        """
        Apply scaling such that the input_image fits to the output_size. Aspect ratio of the image
        could change.

        :param input_image: ITK image
        :param output_size: output size
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return: 
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        current_scale = [(input_size[i] * input_spacing[i]) / (self.output_size[i] * self.output_spacing[i]) for i in range(self.dim)]
        return self.get_scale_transform(current_scale)


class FitFixedAr(ScaleTransformBase):
    def __init__(self, dim, output_size, output_spacing=None):
        super(FitFixedAr, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get(self, **kwargs):
        """
        Apply scaling such that the input_image fits to the output_size. Aspect ratio of the image
        is kept.

        :param input_image: ITK image
        :param output_size: output size
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return: 
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        scale = max([(input_size[i] * input_spacing[i]) / (self.output_size[i] * self.output_spacing[i]) for i in range(self.dim)])
        current_scale = [scale] * self.dim
        return self.get_scale_transform(current_scale)

class InputSpacingToUniformSpacing(ScaleTransformBase):
    def get(self, **kwargs):
        """
        Apply scaling such that the input_image fits to the output_size. Aspect ratio of the image
        could change.

        :param input_image: ITK image
        :param output_size: output size
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        current_scale = [1 / input_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(current_scale)

class UniformSpacingToInputSpacing(ScaleTransformBase):
    def get(self, **kwargs):
        """
        Apply scaling such that the input_image fits to the output_size. Aspect ratio of the image
        could change.

        :param input_image: ITK image
        :param output_size: output size
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        _, input_spacing = self.get_image_size_spacing(**kwargs)
        current_scale = [input_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(current_scale)

class OutputSpacingToUniformSpacing(ScaleTransformBase):
    def __init__(self, dim, output_spacing):
        super(OutputSpacingToUniformSpacing, self).__init__(dim)
        self.output_spacing = output_spacing

    def get(self, **kwargs):
        """
        Apply scaling such that the input_image fits to the output_size. Aspect ratio of the image
        could change.

        :param input_image: ITK image
        :param output_size: output size
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        current_scale = [1 / self.output_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(current_scale)

class UniformSpacingToOutputSpacing(ScaleTransformBase):
    def __init__(self, dim, output_spacing):
        super(UniformSpacingToOutputSpacing, self).__init__(dim)
        self.output_spacing = output_spacing

    def get(self, **kwargs):
        """
        Apply scaling such that the input_image fits to the output_size. Aspect ratio of the image
        could change.

        :param input_image: ITK image
        :param output_size: output size
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        current_scale = [self.output_spacing[i] for i in range(self.dim)]
        return self.get_scale_transform(current_scale)
