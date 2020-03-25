from transformations.spatial.translation import TranslateTransformBase
from utils.landmark.common import get_mean_coords


class Center(TranslateTransformBase):
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
        super(Center, self).__init__(dim, *args, **kwargs)
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
