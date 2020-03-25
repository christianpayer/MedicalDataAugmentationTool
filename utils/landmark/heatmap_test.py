import numpy as np
import SimpleITK as sitk
import utils.geometry
import utils.sitk_image
import utils.sitk_np
import utils.np_image
from utils.landmark.common import Landmark
import utils.landmark.transform


class HeatmapTest(object):
    """
    Used for generating landmark predictions from heatmaps.
    """
    def __init__(self,
                 channel_axis,
                 invert_transformation,
                 interpolator='linear',
                 return_multiple_maxima=False,
                 min_max_value=None,
                 multiple_min_max_value_factor=None,
                 min_max_distance=None):
        """
        Initializer.
        :param channel_axis: The channel axis of the heatmaps of the given numpy arrays.
        :param invert_transformation: If True, invert transformations. Usually, False should be faster and more similarly accurate.
        :param interpolator: The used sitk interpolator that will be used, if invert_transformation is True.
        :param return_multiple_maxima: If True, return multiple non local maxima as landmark coordinates. Else, return only the maximum coordinate.
        :param min_max_value: The minimal value that is considered as a local maximum.
        :param multiple_min_max_value_factor: multiple_min_max_value_factor multiplied with the overall maximum value of the heatmap is considered as the threshold for local maxima.
        :param min_max_distance: min_max_distance (in pixels) is the minimal distance of two local maxima.
        """
        self.channel_axis = channel_axis
        self.invert_transformation = invert_transformation
        self.interpolator = interpolator
        self.return_multiple_maxima = return_multiple_maxima
        self.min_max_value = min_max_value
        self.multiple_min_max_value_factor = multiple_min_max_value_factor
        self.min_max_distance = min_max_distance

    def get_transformed_image_sitk(self, prediction_np, reference_sitk=None, output_spacing=None, transformation=None):
        """
        Returns a list of transformed sitk images from the prediction np array for the given reference image and transformation.
        :param prediction_np: The np array to transform.
        :param reference_sitk: The reference sitk image.
        :param output_spacing: The output spacing of the np array.
        :param transformation: The transformation. If transformation is None, the prediction np array will not be transformed.
        :return: A list of transformed sitk images.
        """
        if transformation is not None:
            predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction_np,
                                                                                  output_spacing=output_spacing,
                                                                                  channel_axis=None,
                                                                                  input_image_sitk=reference_sitk,
                                                                                  transform=transformation,
                                                                                  interpolator=self.interpolator,
                                                                                  output_pixel_type=sitk.sitkFloat32)
        else:
            predictions_np = utils.np_image.split_by_axis(prediction_np, self.channel_axis)
            predictions_sitk = [utils.sitk_np.np_to_sitk(prediction_np) for prediction_np in predictions_np]
        return predictions_sitk

    def get_landmarks(self, prediction_np, reference_sitk=None, output_spacing=None, transformation=None):
        """
        Returns a list of landmarks for the given prediction np array and parameters. The coordinates of the landmarks are the maximum
        of the image, possibly transformed with the transformation parameter.
        :param prediction_np: The np array to transform.
        :param reference_sitk: The reference sitk image.
        :param output_spacing: The output spacing of the np array.
        :param transformation: The transformation. If transformation is None, the prediction np array will not be transformed.
        :return: A list of Landmark objects.
        """
        return [self.get_landmark(image, transformation, reference_sitk, output_spacing) for image in np.rollaxis(prediction_np, self.channel_axis)]

    def get_multiple_maximum_coordinates(self, image):
        """
        Return local maxima of the image. At least one local maximum is returned.
        If the local maximum value absolute_max_value > self.min_max_value, also return other local maxima that are at least self.min_max_distance apart, while
        having a value > absolute_max_value * self.multiple_min_max_value_factor.
        :param image: Heatmap image.
        :return: List of value, coord tuples of local maxima.
        """
        value_coord_pairs = []
        value, coord = utils.np_image.find_quadratic_subpixel_maximum_in_image(image)
        value_coord_pairs.append((value, coord))
        absolute_max_value = value
        if absolute_max_value < self.min_max_value:
            return value_coord_pairs
        image = utils.np_image.draw_sphere(np.copy(image), center=coord, radius=self.min_max_distance, value=0)
        value, coord = utils.np_image.find_quadratic_subpixel_maximum_in_image(image)
        while value > absolute_max_value * self.multiple_min_max_value_factor:
            value_coord_pairs.append((value, coord))
            image = utils.np_image.draw_sphere(np.copy(image), center=coord, radius=self.min_max_distance, value=0)
            value, coord = utils.np_image.find_quadratic_subpixel_maximum_in_image(image)
        return value_coord_pairs

    def get_landmark(self, image, transformation=None, reference_sitk=None, output_spacing=None):
        """
        Returns a single landmark for the given parameters. The coordinates of the landmark are the maximum
        of the image, possibly transformed with the transformation parameter.
        :param image: The np array with a single channel.
        :param reference_sitk: The reference sitk image.
        :param output_spacing: The output spacing of the np array.
        :param transformation: The transformation. If transformation is None, the prediction np array will not be transformed.
        :return: A Landmark object.
        """
        output_spacing = output_spacing or [1] * image.ndim
        if self.return_multiple_maxima:
            landmarks = []
            value_coord_pairs = self.get_multiple_maximum_coordinates(image)
            for value, coord in value_coord_pairs:
                coord = np.flip(coord, axis=0)
                coord *= output_spacing
                coord = utils.landmark.transform.transform_coords(coord, transformation)
                landmarks.append(Landmark(coords=coord, is_valid=True, scale=1, value=value))
            return landmarks
        if transformation is not None:
            if self.invert_transformation:
                # transform prediction back to input image resolution, if specified.
                transformed_sitk = self.get_transformed_image_sitk(image, reference_sitk=reference_sitk, output_spacing=output_spacing, transformation=transformation)
                transformed_np = utils.sitk_np.sitk_to_np_no_copy(transformed_sitk[0])
                value, coord = utils.np_image.find_maximum_in_image(transformed_np)
                coord = np.flip(coord, axis=0)
                coord = coord.astype(np.float32)
                coord *= np.array(reference_sitk.GetSpacing())
            else:
                # search for subpixel accurate maximum in image
                value, coord = utils.np_image.find_quadratic_subpixel_maximum_in_image(image)
                coord = np.flip(coord, axis=0)
                coord *= output_spacing
                coord = utils.landmark.transform.transform_coords(coord, transformation)
        else:
            # just take maximum of image
            value, coord = utils.np_image.find_maximum_in_image(image)
            coord = np.flip(coord, axis=0)

        return Landmark(coords=coord, is_valid=True, scale=1, value=value)
