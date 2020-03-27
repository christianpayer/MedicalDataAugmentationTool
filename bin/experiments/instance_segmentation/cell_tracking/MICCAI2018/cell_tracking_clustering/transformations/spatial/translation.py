import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class TranslateTransformBase(SpatialTransformBase):
    def __init__(self, dim):
        super(TranslateTransformBase, self).__init__(dim)

    def get_flipped(self):
        return [False] * self.dim

    """
    Translation transform for 2D and 3D images.
    """
    def get_translate_transform(self, offset):
        """
        Prepare the translation transform without executing it on an image.

        :param offset: list
            list of floats defining the offset
        :return: SimpleITK.AffineTransform
        """
        assert isinstance(offset, list), 'Offset parameter must be a list of floats, one for each dimension.'
        assert len(offset) in [2, 3], 'Offset must be a list of length 2 for 2D, or 3 for 3D.'

        # print(offset)
        # compute the affine transform
        t = sitk.AffineTransform(len(offset))
        t.Translate(offset)

        # return the transform
        return t

class Fixed(TranslateTransformBase):
    def __init__(self, dim, offset):
        super(Fixed, self).__init__(dim)
        self.current_offset = offset

    def get(self, **kwargs):
        return self.get_translate_transform(self.current_offset)

class Random(TranslateTransformBase):
    def __init__(self, dim, random_offset):
        super(Random, self).__init__(dim)
        self.random_offset = random_offset

    def get(self, **kwargs):
        """
        Apply a random translation, in each dimension specified in the random offset.
        The offset will be computed a uniform distribution 1+[-random_offset, random_offset].

        :param input_image: ITK image
        :param random_offset: float, list of floats
            ranges for uniform random offset in each dimension
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        # translate by individual offset in each dimension
        current_offset = [1.0 + float_uniform(-self.random_offset[i], self.random_offset[i])
                          for i in range(len(self.random_offset))]
        return self.get_translate_transform(current_offset)

class InputCenterToOrigin(TranslateTransformBase):
    def get(self, **kwargs):
        """
        Transform the input center to the origin.
        
        :param input_image: ITK image
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return: 
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        current_offset = [input_size[i] * input_spacing[i] * 0.5
                          for i in range(self.dim)]
        return self.get_translate_transform(current_offset)

class OriginToInputCenter(TranslateTransformBase):
    def get(self, **kwargs):
        """
        Transform the origin to the input center.

        :param input_image: ITK image
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return: 
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        current_offset = [-input_size[i] * input_spacing[i] * 0.5
                               for i in range(self.dim)]
        return self.get_translate_transform(current_offset)

class OutputCenterToOrigin(TranslateTransformBase):
    def __init__(self, dim, output_size, output_spacing=None):
        super(OutputCenterToOrigin, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim
        self.current_offset = [self.output_size[i] * self.output_spacing[i] * 0.5
                               for i in range(self.dim)]

    def get(self, **kwargs):
        return self.get_translate_transform(self.current_offset)

class OriginToOutputCenter(TranslateTransformBase):
    def __init__(self, dim, output_size, output_spacing=None):
        super(OriginToOutputCenter, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim
        self.current_offset = [-self.output_size[i] * self.output_spacing[i] * 0.5
                               for i in range(self.dim)]

    def get(self, **kwargs):
        return self.get_translate_transform(self.current_offset)

