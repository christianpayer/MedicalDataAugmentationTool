
import SimpleITK as sitk
import os
import utils.io.image
import numpy as np
from datasources.base_datasource import BaseDatasource


class ImageDataSource(BaseDatasource):
    def __init__(self,
                 root_location,
                 file_prefix,
                 file_suffix,
                 file_ext,
                 id_dict_preprocessing=None,
                 set_identity_spacing=False,
                 set_zero_origin=True,
                 set_identity_direction=True,
                 preprocessing=None,
                 sitk_pixel_type=sitk.sitkInt16,
                 return_none_if_not_found=False):
        super(ImageDataSource, self).__init__(id_dict_preprocessing=id_dict_preprocessing)
        self.root_location = root_location
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.file_ext = file_ext
        self.set_zero_origin = set_zero_origin
        self.set_identity_direction = set_identity_direction
        self.set_identity_spacing = set_identity_spacing
        self.preprocessing = preprocessing
        self.sitk_pixel_type = sitk_pixel_type
        self.return_none_if_not_found = return_none_if_not_found

    def path_for_id(self, image_id):
        return os.path.join(self.root_location, self.file_prefix + image_id + self.file_suffix + self.file_ext)

    def load_image(self, path):
        try:
            return utils.io.image.read(path, self.sitk_pixel_type)
        except:
            if self.return_none_if_not_found:
                return None
            else:
                raise

    def preprocess(self, image):
        if image is None:
            return image
        if self.set_identity_spacing:
            image.SetSpacing([1] * image.GetDimension())
        if self.set_zero_origin:
            image.SetOrigin([0] * image.GetDimension())
        if self.set_identity_direction:
            image.SetDirection(np.eye(image.GetDimension()).flatten())
        if self.preprocessing is not None:
            image = self.preprocessing(image)
        return image

    def load_and_preprocess(self, image_id):
        image = self.load_image(self.path_for_id(image_id))
        image = self.preprocess(image)
        return image

    def get(self, id_dict, **kwargs):
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        return self.load_and_preprocess(image_id)