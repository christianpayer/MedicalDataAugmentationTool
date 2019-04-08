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

    def get(self, **kwargs):
        """
        Returns the actual sitk transfrom object with the current parameters.
        :param kwargs: Various arguments that may be used by the transformation, e.g., 'image', 'input_size, 'landmarks', etc.
        :return: sitk transform.
        """
        raise NotImplementedError


class Center(LandmarkTransformBase):
    """
    A translation transform that centers the given landmarks at the origin.
    """
    def __init__(self, dim, physical_landmark_coordinates, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param physical_landmark_coordinates: If true, landmark coordinates are in physical units.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(LandmarkTransformBase, self).__init__(dim, *args, **kwargs)
        self.physical_landmark_coordinates = physical_landmark_coordinates

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain 'landmarks' and uses the mean of all landmarks as the center coordinate.
                       If physical_landmark_coordinates is False, must also contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.AffineTransform().
        """
        landmarks = kwargs.get('landmarks')
        current_offset = get_mean_coords(landmarks).tolist()
        if not self.physical_landmark_coordinates:
            input_size, input_spacing, input_direction, input_origin = self.get_image_size_spacing_direction_origin(**kwargs)
            current_offset = self.index_to_physical_point(current_offset, input_origin, input_spacing, input_direction)

        return self.get_translate_transform(self.dim, current_offset)
