import SimpleITK as sitk
import numpy as np

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class TranslateTransformBase(SpatialTransformBase):
    """
    Translation transformation base class.
    """
    def __init__(self, dim, used_dimensions=None, *args, **kwargs):
        """
        Initializer
        :param dim: The dimension.
        :param used_dimensions: Boolean list of which dimension indizes to use for the transformation.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(TranslateTransformBase, self).__init__(dim, *args, **kwargs)
        self.used_dimensions = used_dimensions or [True] * dim
        assert len(self.used_dimensions) == dim, 'Length of used_dimensions must be equal to dim.'

    def get_translate_transform(self, dim, offset):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param offset: List of offsets for each dimension.
        :return: The sitk.AffineTransform().
        """
        assert len(offset) == dim, 'Length of offset must be equal to dim.'

        t = sitk.AffineTransform(dim)
        offset_with_used_dimensions_only = [o if used else 0 for used, o in zip(self.used_dimensions, offset)]
        t.Translate(offset_with_used_dimensions_only)

        return t

    def get(self, **kwargs):
        """
        Returns the actual sitk transfrom object with the current parameters.
        :param kwargs: Various arguments that may be used by the transformation, e.g., 'image', 'input_size, 'landmarks', etc.
        :return: sitk transform.
        """
        raise NotImplementedError


class Fixed(TranslateTransformBase):
    """
    A translation transformation with a fixed offset.
    """
    def __init__(self, dim, offset, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param offset: List of offsets for each dimension.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Fixed, self).__init__(dim, *args, **kwargs)
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
    def __init__(self, dim, random_offset, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param random_offset: List of random offsets per dimension. Random offset is calculated uniformly within [-random_offset[i], random_offset[i]]
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Random, self).__init__(dim, *args, **kwargs)
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


class InputCenterTransformBase(TranslateTransformBase):
    """
    A translation transformation which uses the center of the input image
    """
    def get_input_center(self, **kwargs):
        """
        Returns the input center based on either the parameters defined by the initializer or by **kwargs.
        The function uses the result of self.get_image_size_spacing_direction_origin(**kwargs) to define the output_center for each entry of output_size and output_spacing that is None.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_size, input_spacing, input_direction, input_origin = self.get_image_size_spacing_direction_origin(**kwargs)
        # -1 is important, as it is always the center pixel.
        input_size_half = [(input_size[i] - 1) * 0.5 for i in range(self.dim)]
        return self.index_to_physical_point(input_size_half, input_origin, input_spacing, input_direction)


class InputCenterToOrigin(InputCenterTransformBase):
    """
    A translation transformation which transforms the input image center to the origin.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_center = self.get_input_center(**kwargs)
        return self.get_translate_transform(self.dim, input_center)


class OriginToInputCenter(InputCenterTransformBase):
    """
    A translation transformation which transforms the origin to the input image center.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        input_center = self.get_input_center(**kwargs)
        negative_input_center = [-i for i in input_center]
        return self.get_translate_transform(self.dim, negative_input_center)


class OutputCenterTransformBase(TranslateTransformBase):
    """
    A translation transformation which uses the center of the output image.
    """
    def __init__(self, dim, output_size, output_spacing=None, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(OutputCenterTransformBase, self).__init__(dim, *args, **kwargs)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get_output_center(self, **kwargs):
        """
        Returns the output center based on either the parameters defined by the initializer or by **kwargs.
        The function uses the result of self.get_image_size_spacing(**kwargs) to define the output_center for each entry of output_size and output_spacing that is None.
        :param kwargs: If it contains output_size or output_spacing, use them instead of self.output_size or self.output_spacing. Otherwise, the parameters given to self.get_image_size_spacing(**kwargs).
        :return: List of output center coordinate for each dimension.
        """
        output_size = kwargs.get('output_size', self.output_size)
        output_spacing = kwargs.get('output_spacing', self.output_spacing)
        if not all(output_size):
            # TODO check, if direction or origin are needed
            input_size, input_spacing, input_direction, input_origin = self.get_image_size_spacing_direction_origin(**kwargs)
        else:
            input_size, input_spacing = None, None

        output_center = []
        for i in range(self.dim):
            if output_size[i] is None:
                # -1 is important, as it is always the center pixel.
                output_center.append((input_size[i] - 1) * input_spacing[i] * 0.5)
            else:
                # -1 is important, as it is always the center pixel.
                output_center.append((output_size[i] - 1) * output_spacing[i] * 0.5)
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
    A translation transformation which transforms origin to the output image center.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: These parameters are given to self.get_output_center().
        :return: The sitk.AffineTransform().
        """
        output_center = self.get_output_center(**kwargs)
        negative_output_center = [-o for o in output_center]
        return self.get_translate_transform(self.dim, negative_output_center)


class RandomFactorInput(TranslateTransformBase):
    """
    A translation transform that translates the input image by a random factor, such that it will be cropped.
    The input center should usually be at the origin before this transformation.
    The actual translation value per dimension will be calculated as follows:
    (input_size[i] * input_spacing[i] - self.remove_border[i]) * float_uniform(-self.random_factor[i], self.random_factor[i]) for each dimension.
    """
    def __init__(self, dim, random_factor, remove_border=None, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param random_factor: List of random factors per dimension.
        :param remove_border: List of values that will be subtracted from the input size before calculating the translation value.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(RandomFactorInput, self).__init__(dim, *args, **kwargs)
        self.random_factor = random_factor
        self.remove_border = remove_border or [0] * self.dim

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        # TODO check, if direction or origin are really needed
        input_size, input_spacing, input_direction, input_origin = self.get_image_size_spacing_direction_origin(**kwargs)
        assert np.allclose(input_direction, np.eye(self.dim).flatten()), 'this transformation only works for eye direction, is: ' + input_direction
        assert np.allclose(input_origin, np.zeros(self.dim)), 'this transformation only works for zeros origin, is: ' + input_origin
        max_translation = [input_size[i] * input_spacing[i] - self.remove_border[i] for i in range(self.dim)]
        current_offset = [max_translation[i] * float_uniform(-self.random_factor[i], self.random_factor[i]) for i in range(len(self.random_factor))]
        return self.get_translate_transform(self.dim, current_offset)


class RandomCropInput(TranslateTransformBase):
    """
    A translation transform that crops randomly on the input.
    The input center should usually be at the origin before this transformation.
    The actual translation value per dimension will be calculated as follows:
    (input_size[i] * input_spacing[i] - output_size[i] * output_spacing[i]) * float_uniform(-0.5, 0.5) for each dimension.
    """
    def __init__(self, dim, output_size, output_spacing=None, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(RandomCropInput, self).__init__(dim, *args, **kwargs)
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
        output_size = kwargs.get('output_size', self.output_size)
        output_spacing = kwargs.get('output_spacing', self.output_spacing)
        # TODO check, if direction or origin are really needed
        input_size, input_spacing, input_direction, input_origin = self.get_image_size_spacing_direction_origin(**kwargs)
        assert np.allclose(input_direction, np.eye(self.dim).flatten()), 'this transformation only works for eye direction, is: ' + input_direction
        max_translation = [input_size[i] * input_spacing[i] - output_size[i] * output_spacing[i] for i in range(self.dim)]
        current_offset = [np.maximum(0.0, max_translation[i]) * float_uniform(-0.5, 0.5) for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)


class OriginToBoundingBoxCenter(TranslateTransformBase):
    """
    A translation transformation which transforms the origin to the center of a bounding box.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain 'start' and 'extent' which define the bounding box in physical coordinates.
        :return: The sitk.AffineTransform().
        """
        start = kwargs.get('start')
        extent = kwargs.get('extent')
        current_offset = [-((extent[i] - 1) * 0.5 + start[i]) for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)


class BoundingBoxCenterToOrigin(TranslateTransformBase):
    """
    A translation transformation which transforms the center of a bounding box to the origin.
    """
    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        start = kwargs.get('start')
        extent = kwargs.get('extent')
        current_offset = [(extent[i] - 1) * 0.5 + start[i] for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)


class RandomCropBoundingBox(TranslateTransformBase):
    """
    Performs a crop inside a bounding box.
    """
    def __init__(self, dim, output_size, output_spacing=None, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(RandomCropBoundingBox, self).__init__(dim, *args, **kwargs)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * self.dim

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain 'start' and 'extent' which define the bounding box in physical coordinates.
        :return: The sitk.AffineTransform().
        """
        output_size = kwargs.get('output_size', self.output_size)
        output_spacing = kwargs.get('output_spacing', self.output_spacing)
        extent = kwargs.get('extent', self.output_spacing)
        max_translation = [extent[i] - output_size[i] * output_spacing[i] for i in range(self.dim)]
        current_offset = [np.maximum(0.0, max_translation[i]) * float_uniform(-0.5, 0.5) for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)
