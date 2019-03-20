import os
import SimpleITK as sitk

from datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from datasources.image_datasource import ImageDataSource
from generators.image_generator import ImageGenerator
from transformations.spatial import translation, scale, composite
from transformations.intensity.sitk.smooth import gaussian
from transformations.intensity.sitk.normalize import normalize_robust as normalize_robust_sitk


class Dataset(object):
    def __init__(self,
                 image_size,
                 base_folder='',
                 save_debug_images=False,
                 data_format='channels_first',
                 image_gaussian_blur_sigma=2.0,
                 image_interpolator='linear',
                 normalization_consideration_factors=(0.2, 0.1),
                 debug_folder_prefix='',
                 pad_image=True,
                 crop_image_size=None,
                 additional_scale=None):
        self.image_size = image_size
        self.output_size = self.image_size
        self.base_folder = base_folder
        self.image_base_folder = base_folder
        self.save_debug_images = save_debug_images
        self.data_format = data_format
        self.image_gaussian_blur_sigma = image_gaussian_blur_sigma
        self.image_interpolator = image_interpolator
        self.pad_image = pad_image
        self.crop_image_size = crop_image_size
        self.additional_scale = additional_scale
        if self.additional_scale is None:
            self.additional_scale = [1, 1]
        self.normalization_consideration_factors = normalization_consideration_factors
        self.debug_folder_prefix = debug_folder_prefix
        self.dim = 2

    def image_preprocessing(self, image):
        if self.image_gaussian_blur_sigma > 0:
            image = gaussian(image, self.image_gaussian_blur_sigma)
        if self.crop_image_size is not None:
            image = sitk.Crop(image, self.crop_image_size, self.crop_image_size)
            image.SetOrigin([0] * image.GetDimension())
        if self.pad_image:
            pad_size = image.GetSize()
            pad_size = [int(s / 2) for s in pad_size]
            image = sitk.MirrorPad(image, pad_size, pad_size)
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        image_float = normalize_robust_sitk(image_float, (-1, 1), consideration_factors=self.normalization_consideration_factors)
        return image_float

    def datasources(self):
        datasources_dict = {}
        datasources_dict['image'] = ImageDataSource(self.image_base_folder, 't', '', '.tif',
                                                    set_identity_spacing=True,
                                                    set_zero_origin=True,
                                                    set_identity_direction=True,
                                                    preprocessing=self.image_preprocessing,
                                                    sitk_pixel_type=sitk.sitkUInt16)
        return datasources_dict

    def data_generators(self):
        data_generators_dict = {}
        data_generators_dict['image'] = ImageGenerator(self.dim, self.image_size, interpolator=self.image_interpolator, data_format=self.data_format, return_zeros_if_not_found=True, resample_default_pixel_value=-1)
        return data_generators_dict

    def data_generator_sources(self):
        return {'image': {'image': 'image'}}


    def image_transformation_val(self):
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.Fit(self.dim, self.image_size),
                                    scale.Fixed(self.dim, self.additional_scale),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)])

    def dataset_val_single_frame(self):
        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        generators_val = self.data_generators()
        image_transformation = self.image_transformation_val()

        dataset_val = ReferenceTransformationDataset(dim=self.dim,
                                                     reference_datasource_keys={'image': 'image'},
                                                     reference_transformation=image_transformation,
                                                     datasources=sources,
                                                     data_generators=generators_val,
                                                     data_generator_sources=generator_sources,
                                                     iterator=None,
                                                     all_generators_post_processing=None,
                                                     debug_image_folder=os.path.join(self.debug_folder_prefix, 'debug_val') if self.save_debug_images else None)

        return dataset_val
