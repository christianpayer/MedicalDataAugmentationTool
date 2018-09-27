import os

import numpy as np
import SimpleITK as sitk

from datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image, distance_transform
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from transformations.intensity.np.normalize import normalize_robust


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
                 label_gaussian_sigma=1.0,
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
        :param label_gaussian_sigma: Sigma value for label smoothing.
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.modality = modality
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.image_base_folder = os.path.join(self.base_folder, modality + '_mha')
        self.setup_base_folder = os.path.join(self.base_folder, 'setup')
        self.landmarks_file = os.path.join(self.setup_base_folder, modality + '_seg_center_rai.csv')

        if modality == 'ct':
            self.postprocessing_random = self.intensity_postprocessing_ct_random
            self.postprocessing = self.intensity_postprocessing_ct
        else:  # if modality == 'mr':
            self.postprocessing_random = self.intensity_postprocessing_mr_random
            self.postprocessing = self.intensity_postprocessing_mr
        if cv > 0:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join(modality + '_cv', str(cv)))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.test_file = os.path.join(self.cv_folder, 'test.txt')
        else:
            self.train_file = os.path.join(self.setup_base_folder, modality + '_train_all.txt')
            self.test_file = os.path.join(self.setup_base_folder, modality + '_test_all.txt')

    def datasources(self):
        """
        Returns the data sources that load data.
        {
        'image:' CachedImageDataSource that loads the image files.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        'mask:' CachedImageDataSource that loads the groundtruth labels.
        }
        :return: A dict of data sources.
        """
        preprocessing = lambda image: gaussian_sitk(image, self.input_gaussian_sigma)
        image_datasource = CachedImageDataSource(self.image_base_folder, '', '_image', '.mha', preprocessing=preprocessing)
        landmark_datasource = LandmarkDataSource(self.landmarks_file, 1, self.dim)
        mask_datasource = CachedImageDataSource(self.image_base_folder, '', '_label_sorted', '.mha', sitk_pixel_type=sitk.sitkUInt8)
        return {'image': image_datasource,
                'landmarks': landmark_datasource,
                'mask': mask_datasource}

    def data_generators(self, image_post_processing, mask_post_processing):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, data_format=self.data_format)
        mask_image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='nearest', post_processing_np=mask_post_processing, data_format=self.data_format)
        return {'data': image_generator,
                'mask': mask_image_generator}

    def data_generator_sources(self):
        """
        Returns a dict that defines the connection between datasources and datagenerator parameters for their get() function.
        :return: A dict.
        """
        return {'data': {'image': 'image'},
                'mask': {'image': 'mask'}}

    def split_labels(self, image):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :return: The one-hot encoded image.
        """
        split = split_label_image(np.squeeze(image, 0), list(range(8)), np.uint8)
        split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
        smoothed = np.stack(split_smoothed, 0)
        image_smoothed = np.argmax(smoothed, axis=0)
        split = split_label_image(image_smoothed, list(range(8)), np.uint8)
        return np.stack(split, 0)

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

    def intensity_postprocessing_mr_random(self, image):
        """
        Intensity postprocessing for MR input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(random_shift=0.2,
                               random_scale=0.4,
                               clamp_min=-1.0)(image)

    def intensity_postprocessing_mr(self, image):
        """
        Intensity postprocessing for MR input.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(clamp_min=-1.0)(image)

    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [landmark.Center(self.dim, False),
                                    translation.Random(self.dim, [20, 20, 20]),
                                    rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    deformation.Output(self.dim, [8, 8, 8], 15, self.image_size, self.image_spacing)]
                                   )

    def spatial_transformation(self):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [landmark.Center(self.dim, False),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing)]
                                   )

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_file, random=True, keys=['image_id'])
        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        generators = self.data_generators(self.postprocessing_random, self.split_labels)
        reference_transformation = self.spatial_transformation_augmented()

        return ReferenceTransformationDataset(dim=self.dim,
                                              reference_datasource_keys={'image': 'image',
                                                                         'landmarks': 'landmarks'},
                                              reference_transformation=reference_transformation,
                                              datasources=sources,
                                              data_generators=generators,
                                              data_generator_sources=generator_sources,
                                              iterator=iterator,
                                              debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'])
        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        generators = self.data_generators(self.postprocessing, self.split_labels)
        reference_transformation = self.spatial_transformation()

        if self.cv == 0:
            del sources['mask']
            del generator_sources['mask']
            del generators['mask']

        return ReferenceTransformationDataset(dim=self.dim,
                                              reference_datasource_keys={'image': 'image',
                                                                         'landmarks': 'landmarks'},
                                              reference_transformation=reference_transformation,
                                              datasources=sources,
                                              data_generators=generators,
                                              data_generator_sources=generator_sources,
                                              iterator=iterator,
                                              debug_image_folder='debug_val' if self.save_debug_images else None)
