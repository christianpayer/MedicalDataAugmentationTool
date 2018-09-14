import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class TranslateTransformBase(SpatialTransformBase):
    """
    Translation transformation base class.
    """
    @staticmethod
    def get_translate_transform(dim, offset):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param offset: List of offsets for each dimension.
        :return: The sitk.AffineTransform().
        """
        assert len(offset) == dim, 'Length of offset must be equal to dim.'

        t = sitk.AffineTransform(dim)
        t.Translate(offset)

        return t


class Fixed(TranslateTransformBase):
    """
    A translation transformation with a fixed offset.
    """
    def __init__(self, dim, offset):
        """
        Initializer.
        :param dim: The dimension.
        :param offset: List of offsets for each dimension.
        """
        super(Fixed, self).__init__(dim)
        self.current_offset = offset

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        return self.get_translate_transform(self.dim, self.current_offset)


class Random(TranslateTransformBase):
    """
    A translation transformation with a random offset.
    """
    def __init__(self, dim, random_offset):
        """
        Initializer.
        :param dim: The dimension.
        :param random_offset: List of random offsets per dimension. Random offset is calculated uniformly within [-random_offset[i], random_offset[i]]
        """
        super(Random, self).__init__(dim)
        self.random_offset = random_offset

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        current_offset = [float_uniform(-self.random_offset[i], self.random_offset[i])
                          for i in range(len(self.random_offset))]
        return self.get_translate_transform(self.dim, current_offset)


class InputCenterToOrigin(TranslateTransformBase):
    """
    A translation transformation which transforms the input image center to the origin.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        # -1 is important, as it is always the center pixel.
        current_offset = [(input_size[i] - 1) * input_spacing[i] * 0.5
                          for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)


class OriginToInputCenter(TranslateTransformBase):
    """
    A translation transformation which transforms the origin to the the input image center.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        # -1 is important, as it is always the center pixel.
        current_offset = [-(input_size[i] - 1) * input_spacing[i] * 0.5
                          for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)


class OutputCenterTransformBase(TranslateTransformBase):
    """
    A translation transformation which transforms the output image.
    """
    def __init__(self, dim, output_size, output_spacing=None):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        """
        super(OutputCenterTransformBase, self).__init__(dim)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get_output_center(self, **kwargs):
        """
        Returns the output center based on either the parameters defined by the initializer or by **kwargs.
        The function uses the response of self.get_image_size_spacing(**kwargs) to define the output_center for each entry of output_size and output_spacing that is None.
        :param kwargs: Parameters given to self.get_image_size_spacing(**kwargs)
        :return: List of output center coordinate for each dimension.
        """
        if not all(self.output_size):
            input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        else:
            input_size, input_spacing = None, None

        output_center = []
        for i in range(self.dim):
            if self.output_size[i] is None:
                # -1 is important, as it is always the center pixel.
                output_center.append((input_size[i] - 1) * input_spacing[i] * 0.5)
            else:
                # -1 is important, as it is always the center pixel.
                output_center.append((self.output_size[i] - 1) * self.output_spacing[i] * 0.5)
        return output_center


class OutputCenterToOrigin(OutputCenterTransformBase):
    """
    A translation transformation which transforms the output image center to the origin.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: These parameters are given to self.get_output_center().
        :return: The sitk.AffineTransform().
        """
        output_center = self.get_output_center(**kwargs)
        return self.get_translate_transform(self.dim, output_center)


class OriginToOutputCenter(OutputCenterTransformBase):
    """
    A translation transformation which transforms origin to the the output image center.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: These parameters are given to self.get_output_center().
        :return: The sitk.AffineTransform().
        """
        output_center = self.get_output_center(**kwargs)
        return self.get_translate_transform(self.dim, [-o for o in output_center])


class RandomFactorInput(TranslateTransformBase):
    """
    A translation transform that translates the input image by a random factor, such that it will be cropped.
    The actual translation value per dimension will be calculated as follows:
    (input_size[i] * input_spacing[i] - remove_border[i]) * float_uniform(-self.random_factor[i], self.random_factor[i]) for each dimension.
    """
    def __init__(self, dim, random_factor, remove_border=None):
        """
        Initializer.
        :param dim: The dimension.
        :param random_factor: List of random factors per dimension.
        :param remove_border: List of values that will be subtracted from the input size before calculating the translation value.
        """
        super(RandomFactorInput, self).__init__(dim)
        self.random_factor = random_factor
        self.remove_border = remove_border

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)
        remove_border = self.remove_border
        if remove_border is None:
            remove_border = [0] * self.dim

        current_offset = [(input_size[i] * input_spacing[i] - remove_border[i]) * float_uniform(-self.random_factor[i], self.random_factor[i])
                          for i in range(len(self.random_factor))]
        return self.get_translate_transform(self.dim, current_offset)
