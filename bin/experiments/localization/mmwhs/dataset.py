import os

import numpy as np
import SimpleITK as sitk

from datasets.graph_dataset import GraphDataset
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGenerator, LandmarkGeneratorHeatmap
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.intensity.sitk.shift_scale_clamp import ShiftScaleClamp as ShiftScaleClampSitk
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image, smooth_label_images
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.normalize import normalize_robust
from transformations.intensity.np.gamma import change_gamma_unnormalized
from utils.random import float_uniform


class Dataset(object):
    """
    The dataset that processes files from the MMWHS challenge.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 base_folder,
                 cv,
                 modality,
                 input_gaussian_sigma=1.0,
                 heatmap_sigma=1.5,
                 data_format='channels_first',
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param image_spacing: Network input image spacing.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3). Or 0 if full training/testing.
        :param modality: Either 'ct' or 'mr'.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.modality = modality
        self.input_gaussian_sigma = input_gaussian_sigma
        self.heatmap_sigma = heatmap_sigma
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.image_base_folder = os.path.join(self.base_folder, modality + '_mha')
        self.setup_base_folder = os.path.join(self.base_folder, 'setup')
        self.landmarks_file = os.path.join(self.setup_base_folder, modality + '_seg_center_rai_w_spacing.csv')

        if modality == 'ct':
            self.preprocessing = self.intensity_preprocessing_ct
            self.postprocessing_random = self.intensity_postprocessing_ct_random
            self.postprocessing = self.intensity_postprocessing_ct
            self.image_default_pixel_value = -1024
        else:  # if modality == 'mr':
            self.preprocessing = self.intensity_preprocessing_mr
            self.postprocessing_random = self.intensity_postprocessing_mr_random
            self.postprocessing = self.intensity_postprocessing_mr
            self.image_default_pixel_value = 0
        if cv > 0:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join(modality + '_cv', str(cv)))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.test_file = os.path.join(self.cv_folder, 'test.txt')
        else:
            self.train_file = os.path.join(self.setup_base_folder, modality + '_train_all.txt')
            self.test_file = os.path.join(self.setup_base_folder, modality + '_test_all.txt')

    def datasources(self, iterator, cached):
        """
        Returns the data sources that load data.
        {
        'image:' CachedImageDataSource that loads the image files.
        'labels:' CachedImageDataSource that loads the groundtruth labels.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param iterator: The dataset iterator.
        :param cached: If true, use CachedImageDataSource, else ImageDataSource.
        :return: A dict of data sources.
        """
        datasources_dict = {}
        image_data_source = CachedImageDataSource if cached else ImageDataSource
        datasources_dict['image'] = image_data_source(self.image_base_folder, '', '_image', '.mha', set_zero_origin=True, set_identity_direction=True, set_identity_spacing=False, sitk_pixel_type=sitk.sitkInt16, preprocessing=self.preprocessing, name='image', parents=[iterator])
        datasources_dict['landmarks'] = LandmarkDataSource(self.landmarks_file, 1, self.dim, name='landmarks', parents=[iterator])
        return datasources_dict

    def data_generators(self, datasources, transformation, image_post_processing):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param datasources: datasources dict.
        :param transformation: transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, data_format=self.data_format, resample_default_pixel_value=self.image_default_pixel_value, name='image', parents=[datasources['image'], transformation])
        landmark_generator = LandmarkGeneratorHeatmap(self.dim, self.image_size, self.image_spacing, self.heatmap_sigma, 1, True, name='landmarks', parents=[datasources['landmarks'], transformation])
        return {'image': image_generator,
                'landmarks': landmark_generator}

    def intensity_preprocessing_ct(self, image):
        """
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        image = ShiftScaleClampSitk(clamp_min=-1024)(image)
        return gaussian_sitk(image, self.input_gaussian_sigma)

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               random_shift=0.2,
                               random_scale=0.2,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_postprocessing_ct(self, image):
        """
        Intensity postprocessing for CT input.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_preprocessing_mr(self, image):
        """
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        return gaussian_sitk(image, self.input_gaussian_sigma)

    def intensity_postprocessing_mr_random(self, image):
        """
        Intensity postprocessing for MR input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        image = change_gamma_unnormalized(image, float_uniform(0.5, 1.5))
        image = normalize_robust(image, consideration_factors=(0.1, 0.1))
        return ShiftScaleClamp(random_shift=0.6,
                               random_scale=0.6,
                               clamp_min=-1.0)(image)

    def intensity_postprocessing_mr(self, image):
        """
        Intensity postprocessing for MR input.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image, consideration_factors=(0.1, 0.1))
        return ShiftScaleClamp(clamp_min=-1.0)(image)

    def spatial_transformation_augmented(self, datasources):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.extend([translation.Random(self.dim, [20, 20, 20]),
                                    rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    deformation.Output(self.dim, [6, 6, 6], 15, self.image_size, self.image_spacing)
                                    ])
        return composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)

    def spatial_transformation(self, datasources):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing))
        return composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_file, random=True, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, True)
        reference_transformation = self.spatial_transformation_augmented(sources)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing_random)

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, False)
        reference_transformation = self.spatial_transformation(sources)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing)

        if self.cv == 0:
            del sources['landmarks']
            del generators['landmarks']

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)
