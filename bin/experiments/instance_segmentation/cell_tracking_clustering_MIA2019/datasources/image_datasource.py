
import SimpleITK as sitk
import os
import utils.io.image
import numpy as np
from datasources.datasource_base import DataSourceBase


class ImageDataSource(DataSourceBase):
    """
    DataSource used for loading sitk images. Uses id_dict['image_id'] as image path and returns the sitk_image at the given path.
    Preprocesses the path as follows: file_path_to_load = os.path.join(root_location, file_prefix + id_dict['image_id'] + file_suffix + file_ext)
    """
    def __init__(self,
                 root_location,
                 file_prefix='',
                 file_suffix='',
                 file_ext='.mha',
                 set_identity_spacing=False,
                 set_zero_origin=True,
                 set_identity_direction=True,
                 round_spacing_precision=None,
                 preprocessing=None,
                 sitk_pixel_type=sitk.sitkInt16,
                 return_none_if_not_found=False,
                 *args, **kwargs):
        """
        Initializer.
        :param root_location: Root path, where the images will be loaded from.
        :param file_prefix: Prefix of the file path.
        :param file_suffix: Suffix of the file path.
        :param file_ext: Extension of the file path.
        :param id_dict_preprocessing: Function that will be called for preprocessing a given id_dict.
        :param set_identity_spacing: If true, the spacing of the sitk image will be set to 1 for every dimension.
        :param set_zero_origin: If true, the origin of the sitk image will be set to 0 for every dimension.
        :param set_identity_direction: If true, the direction of the sitk image will be set to 1 for every dimension.
        :param round_spacing_precision: If > 0, spacing will be rounded to this precision (as in round(x, round_spacing_origin_direction))
        :param preprocessing: Function that will be called for preprocessing a loaded sitk image, i.e., sitk_image = preprocessing(sitk_image)
        :param sitk_pixel_type: sitk pixel type to which the loaded image will be converted to.
        :param return_none_if_not_found: If true, instead of raising an exception, None will be returned, if the image at the given path could not be loaded.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(ImageDataSource, self).__init__(*args, **kwargs)
        self.root_location = root_location
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.file_ext = file_ext
        self.set_zero_origin = set_zero_origin
        self.set_identity_direction = set_identity_direction
        self.set_identity_spacing = set_identity_spacing
        self.round_spacing_precision = round_spacing_precision
        self.preprocessing = preprocessing
        self.sitk_pixel_type = sitk_pixel_type
        self.return_none_if_not_found = return_none_if_not_found

    def path_for_id(self, image_id):
        """
        Generates the path for a given image_id. returns os.path.join(root_location, file_prefix + id_dict['image_id'] + file_suffix + file_ext)
        :param image_id: The image_id.
        :return: The file path for the given image_id.
        """
        return os.path.join(self.root_location, self.file_prefix + image_id + self.file_suffix + self.file_ext)

    def load_image(self, path):
        """
        Loads an image from a given path. Throws an exception, if the image could not be loaded. If return_none_if_not_found is True, instead of throwing
        an exception, None will be returned in case the image could not be loaded.
        :param path: The file path.
        :return: The loaded sitk image.
        """
        try:
            return utils.io.image.read(path, self.sitk_pixel_type)
        except:
            if self.return_none_if_not_found:
                return None
            else:
                raise

    def preprocess(self, image):
        """
        Processes the loaded image based on the given parameters of __init__(), i.e.,
        set_identity_spacing, set_zero_origin, set_identity_direction, preprocessing
        :param image: The loaded sitk image.
        :return: The processed sitk image.
        """
        if image is None:
            return image
        if self.set_identity_spacing:
            image.SetSpacing([1] * image.GetDimension())
        if self.set_zero_origin:
            image.SetOrigin([0] * image.GetDimension())
        if self.set_identity_direction:
            image.SetDirection(np.eye(image.GetDimension()).flatten())
        if self.round_spacing_precision is not None:
            image.SetSpacing([round(x, self.round_spacing_precision) for x in image.GetSpacing()])
        if self.preprocessing is not None:
            image = self.preprocessing(image)
        return image

    def load_and_preprocess(self, image_id):
        """
        Loads an image for a given image_id and performs additional processing.
        :param image_id: The image_id.
        :return: The loaded and processed sitk image.
        """
        image = self.load_image(self.path_for_id(image_id))
        image = self.preprocess(image)
        return image

    def get(self, id_dict):
        """
        Loads and processes an image for a given id_dict. Returns the sitk image.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as the path for loading the sitk image.
        :return: The loaded and processed sitk image.
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        return self.load_and_preprocess(image_id)