
import os
import numpy as np
import SimpleITK as sitk
from datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from datasources.image_datasource import ImageDataSource
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGeneratorHeatmap, LandmarkGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, deformation
from transformations.intensity.np.normalize import normalize_robust
from utils.sitk_image import reduce_dimension


class Dataset(object):
    """
    The dataset that processes files from hand xray dataset.
    """
    def __init__(self,
                 image_size,
                 heatmap_size,
                 num_landmarks,
                 sigma,
                 base_folder,
                 cv,
                 data_format,
                 save_debug_images):
        """
        Initializer.
        :param image_size: Network input image size.
        :param heatmap_size: Network output image size.
        :param num_landmarks: The number of landmarks.
        :param sigma: The heatmap sigma.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3).
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.downsampling_factor = self.image_size[0] / self.heatmap_size[0]
        self.num_landmarks = num_landmarks
        self.sigma = sigma
        self.base_folder = base_folder
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 2
        self.image_base_folder = os.path.join(self.base_folder, 'images')
        self.setup_base_folder = os.path.join(self.base_folder, 'setup')
        self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'cv_reduced', 'set' + str(cv), 'train.txt')
        self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'cv_reduced', 'set' + str(cv), 'test.txt')
        self.point_list_file_name = os.path.join(self.setup_base_folder, 'all.csv')

    def data_sources(self, cached, image_extension='.mha'):
        """
        Returns the data sources that load data.
        {
        'image_datasource:' ImageDataSource that loads the image files.
        'landmarks_datasource:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param cached: If true, use a CachedImageDataSource instead of an ImageDataSource.
        :param image_extension: The image extension of the input data.
        :return: A dict of data sources.
        """
        if cached:
            image_datasource = CachedImageDataSource(self.image_base_folder,
                                                     '',
                                                     '',
                                                     image_extension,
                                                     preprocessing=reduce_dimension,
                                                     set_identity_spacing=True,
                                                     cache_maxsize=16384)
        else:
            image_datasource = ImageDataSource(self.image_base_folder,
                                               '',
                                               '',
                                               image_extension,
                                               preprocessing=reduce_dimension,
                                               set_identity_spacing=True)
        landmarks_datasource = LandmarkDataSource(self.point_list_file_name,
                                                  self.num_landmarks,
                                                  self.dim)
        return {'image_datasource': image_datasource,
                'landmarks_datasource': landmarks_datasource}

    def data_generators(self, image_post_processing_np):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param image_post_processing_np: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim,
                                         self.image_size,
                                         post_processing_np=image_post_processing_np,
                                         interpolator='linear',
                                         resample_default_pixel_value=0,
                                         data_format=self.data_format,
                                         resample_sitk_pixel_type=sitk.sitkFloat32,
                                         np_pixel_type=np.float32)
        if self.downsampling_factor == 1:
            heatmap_post_transformation = None
        else:
            heatmap_post_transformation = scale.Fixed(self.dim, self.downsampling_factor)
        landmark_generator = LandmarkGeneratorHeatmap(self.dim,
                                                      self.heatmap_size,
                                                      self.sigma,
                                                      scale_factor=1.0,
                                                      normalize_center=True,
                                                      data_format=self.data_format,
                                                      post_transformation=heatmap_post_transformation)
        return {'image': image_generator,
                'landmarks': landmark_generator}

    def data_generator_sources(self):
        """
        Returns a dict that defines the connection between datasources and datagenerator parameters for their get() function.
        :return: A dict.
        """
        return {'image': {'image': 'image_datasource'},
                'landmarks': {'landmarks': 'landmarks_datasource'}}

    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.FitFixedAr(self.dim, self.image_size),
                                    translation.Random(self.dim, [10, 10]),
                                    rotation.Random(self.dim, [0.2, 0.2]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size),
                                    deformation.Output(self.dim, [5, 5], 20, self.image_size)
                                    ]
                                   )

    def spatial_transformation(self):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.FitFixedAr(self.dim, self.image_size),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)]
                                   )

    def intensity_postprocessing_augmented(self, image):
        """
        Intensity postprocessing. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        normalized = normalize_robust(image)
        return ShiftScaleClamp(random_shift=0.15,
                               random_scale=0.15)(normalized)

    def intensity_postprocessing(self, image):
        """
        Intensity postprocessing.
        :param image: The np input image.
        :return: The processed image.
        """
        normalized = normalize_robust(image)
        return normalized

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        data_sources = self.data_sources(True)
        data_generator_sources = self.data_generator_sources()
        data_generators = self.data_generators(self.intensity_postprocessing_augmented)
        image_transformation = self.spatial_transformation_augmented()
        iterator = IdListIterator(self.train_id_list_file_name,
                                  random=True,
                                  keys=['image_id'])
        dataset = ReferenceTransformationDataset(dim=self.dim,
                                                 reference_datasource_keys={'image': 'image_datasource'},
                                                 reference_transformation=image_transformation,
                                                 datasources=data_sources,
                                                 data_generators=data_generators,
                                                 data_generator_sources=data_generator_sources,
                                                 iterator=iterator,
                                                 debug_image_folder='debug_train' if self.save_debug_images else None)
        return dataset

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        data_sources = self.data_sources(False)
        data_generator_sources = self.data_generator_sources()
        data_generators = self.data_generators(self.intensity_postprocessing)
        image_transformation = self.spatial_transformation()
        iterator = IdListIterator(self.val_id_list_file_name,
                                  random=False,
                                  keys=['image_id'])
        dataset = ReferenceTransformationDataset(dim=self.dim,
                                                 reference_datasource_keys={'image': 'image_datasource'},
                                                 reference_transformation=image_transformation,
                                                 datasources=data_sources,
                                                 data_generators=data_generators,
                                                 data_generator_sources=data_generator_sources,
                                                 iterator=iterator,
                                                 debug_image_folder='debug_val' if self.save_debug_images else None)
        return dataset
