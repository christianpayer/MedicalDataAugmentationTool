import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.landmark.common import get_mean_coords


class LandmarkTransformBase(SpatialTransformBase):
    """
    Landmark transformation base class.
    TODO: merge with TranslateTransformBase()
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


class Center(LandmarkTransformBase):
    """
    A translation transform that centers the given landmarks at the origin.
    """
    def __init__(self, dim, uniform_spacing):
        """
        Initializer.
        :param dim: The dimension.
        :param uniform_spacing: If true, uses uniform spacing of 1 mm, otherwise, uses spacing of input image.
        """
        super(LandmarkTransformBase, self).__init__(dim)
        self.uniform_spacing = uniform_spacing

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain 'landmarks' and uses the mean of all landmarks as the center coordinate.
                       If uniform_spacing is False, must also contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        landmarks = kwargs.get('landmarks')
        mean_coords = get_mean_coords(landmarks)
        if self.uniform_spacing:
            input_spacing = [1] * self.dim
        else:
            _, input_spacing = self.get_image_size_spacing(**kwargs)

        current_offset = [mean_coords[i] * input_spacing[i] for i in range(self.dim)]
        return self.get_translate_transform(self.dim, current_offset)
