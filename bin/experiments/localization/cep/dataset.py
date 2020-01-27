
import os

import SimpleITK as sitk
import numpy as np

from datasets.graph_dataset import GraphDataset
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGenerator
from graph.node import LambdaNode
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.intensity.sitk.smooth import gaussian
from transformations.spatial import translation, scale, composite, rotation, deformation
from utils.landmark.common import get_mean_landmark_list
from utils.random import bool_bernoulli


class Dataset(object):
    """
    The dataset that processes files from hand xray dataset.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 heatmap_size,
                 num_landmarks,
                 base_folder,
                 cv=-1,
                 image_gaussian_sigma=0.15,
                 landmark_source='challenge',
                 data_format='channels_first',
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param image_spacing: Network input image spacing.
        :param heatmap_size: Network output image size.
        :param num_landmarks: The number of landmarks.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3).
        :param image_gaussian_sigma: Gaussian sigma in mm for preprocessing the input image.
        :param landmark_source: Select the landmark file to use:
                                'challenge': annotation used at the 2015 challenge.
                                'senior': annotation of the senior radiologist used in the 2016 paper from Lindner et al.
                                'junior': annotation of the junior radiologist used in the 2016 paper from Lindner et al.
                                'mean': mean landmark annotation of senior and junior radiologist used in the 2016 paper from Lindner et al.
                                'random': randomly use either senior or junior annotation when calling dataset.get_next().
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.heatmap_size = heatmap_size
        self.downsampling_factor = self.image_size[0] / self.heatmap_size[0]
        self.num_landmarks = num_landmarks
        self.base_folder = base_folder
        self.cv = cv
        self.image_gaussian_sigma = image_gaussian_sigma
        self.landmark_source = landmark_source
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 2
        self.image_base_folder = os.path.join(self.base_folder, 'images')
        self.setup_base_folder = os.path.join(self.base_folder, 'setup')

        if cv > 0:
            self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'cv', 'set{}'.format(cv), 'train.txt')
            self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'cv', 'set{}'.format(cv), 'val.txt')
        else:
            self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'train.txt')
            self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'test_all.txt')
        self.junior_point_list_file_name = os.path.join(self.setup_base_folder, 'all_junior.csv')
        self.senior_point_list_file_name = os.path.join(self.setup_base_folder, 'all_senior.csv')
        self.challenge_point_list_file_name = os.path.join(self.setup_base_folder, 'all_challenge.csv')

    def image_preprocessing(self, image):
        if self.image_gaussian_sigma > 0.0:
            return gaussian(image, self.image_gaussian_sigma)
        else:
            return image

    def data_sources(self, cached, iterator, image_extension='.nii.gz'):
        """
        Returns the data sources that load data.
        {
        'image_datasource:' ImageDataSource that loads the image files.
        'landmarks_datasource:' LandmarkDataSource that loads the landmark coordinates.
        'junior_landmarks_datasource',
        'senior_landmarks_datasource',
        'challenge_landmarks_datasource',
        'mean_landmarks_datasource',
        'random_landmarks_datasource': other LandmarkDataSources for different groundtruths.
        }
        :param cached: If true, use a CachedImageDataSource instead of an ImageDataSource.
        :param iterator: The iterator that is used as the parent node.
        :param image_extension: The image extension of the input data.
        :return: A dict of data sources.
        """
        if cached:
            image_datasource = CachedImageDataSource(self.image_base_folder,
                                                     '',
                                                     '',
                                                     image_extension,
                                                     preprocessing=self.image_preprocessing,
                                                     set_identity_spacing=False,
                                                     cache_maxsize=16384,
                                                     parents=[iterator],
                                                     name='image_datasource')
        else:
            image_datasource = ImageDataSource(self.image_base_folder,
                                               '',
                                               '',
                                               image_extension,
                                               preprocessing=self.image_preprocessing,
                                               set_identity_spacing=False,
                                               parents=[iterator],
                                               name='image_datasource')
        landmark_datasources = self.landmark_datasources(iterator)
        return {'image_datasource': image_datasource, **landmark_datasources}

    def landmark_datasources(self, iterator):
        """
        Returns all LandmarkDataSources as a dictionary. The entry 'landmark_data_source' is set to the value set in self.landmark_source.
        Other entries: 'junior_landmarks_datasource',
                       'senior_landmarks_datasource',
                       'challenge_landmarks_datasource',
                       'mean_landmarks_datasource',
                       'random_landmarks_datasource'.
        :param iterator: The iterator that is used as the parent node.
        :return: Dictionary of all LandmarkDataSources.
        """
        senior_landmarks_datasource = LandmarkDataSource(self.senior_point_list_file_name,
                                                         self.num_landmarks,
                                                         self.dim,
                                                         parents=[iterator],
                                                         name='senior_landmarks_datasource')
        junior_landmarks_datasource = LandmarkDataSource(self.junior_point_list_file_name,
                                                         self.num_landmarks,
                                                         self.dim,
                                                         parents=[iterator],
                                                         name='junior_landmarks_datasource')
        challenge_landmarks_datasource = LandmarkDataSource(self.challenge_point_list_file_name,
                                                            self.num_landmarks,
                                                            self.dim,
                                                            parents=[iterator],
                                                            name='challenge_landmarks_datasource')
        mean_landmarks_datasource = LambdaNode(lambda junior, senior: get_mean_landmark_list(junior, senior), parents=[junior_landmarks_datasource, senior_landmarks_datasource], name='mean_landmarks_datasource')
        random_landmarks_datasource = LambdaNode(lambda junior, senior: senior if bool_bernoulli(0.5) else junior, parents=[junior_landmarks_datasource, senior_landmarks_datasource], name='random_landmarks_datasource')
        if self.landmark_source == 'senior':
            landmark_datasource =  senior_landmarks_datasource
        elif self.landmark_source == 'junior':
            landmark_datasource =  junior_landmarks_datasource
        elif self.landmark_source == 'challenge':
            landmark_datasource = challenge_landmarks_datasource
        elif self.landmark_source == 'mean':
            landmark_datasource = mean_landmarks_datasource
        elif self.landmark_source == 'random':
            landmark_datasource = random_landmarks_datasource

        return {'junior_landmarks_datasource': junior_landmarks_datasource,
                'senior_landmarks_datasource': senior_landmarks_datasource,
                'challenge_landmarks_datasource': challenge_landmarks_datasource,
                'mean_landmarks_datasource': mean_landmarks_datasource,
                'random_landmarks_datasource': random_landmarks_datasource,
                'landmark_datasource': landmark_datasource}


    def data_generators(self, data_sources, transformation, image_post_processing_np):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param data_sources: The data_sources dictionary.
        :param transformation: The used transformation.
        :param image_post_processing_np: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim,
                                         self.image_size,
                                         self.image_spacing,
                                         post_processing_np=image_post_processing_np,
                                         interpolator='linear',
                                         resample_default_pixel_value=0,
                                         data_format=self.data_format,
                                         resample_sitk_pixel_type=sitk.sitkFloat32,
                                         np_pixel_type=np.float32,
                                         parents=[data_sources['image_datasource'], transformation], name='image')
        if self.downsampling_factor == 1:
            heatmap_post_transformation = None
        else:
            heatmap_post_transformation = scale.Fixed(self.dim, self.downsampling_factor)
        landmark_generator = LandmarkGenerator(self.dim,
                                               self.heatmap_size,
                                               self.image_spacing,
                                               #sigma=1.0,
                                               #scale_factor=1.0,
                                               #normalize_center=True,
                                               data_format=self.data_format,
                                               post_transformation=heatmap_post_transformation,
                                               min_max_transformation_distance=30,
                                               parents=[data_sources['landmark_datasource'], transformation], name='landmarks')
        return {'image': image_generator,
                'landmarks': landmark_generator}

    def spatial_transformation_augmented(self, data_sources):
        """
        The spatial image transformation with random augmentation.
        :param data_sources: The data_sources dictionary.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    translation.Random(self.dim, [10, 10]),
                                    rotation.Random(self.dim, [0.25]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.2, 0.2]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    deformation.Output(self.dim, [5, 5], 10, self.image_size, self.image_spacing)
                                    ],
                                   kwparents={'image': data_sources['image_datasource']}, name='image')

    def spatial_transformation(self, data_sources):
        """
        The spatial image transformation without random augmentation.
        :param data_sources: The data_sources dictionary.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing)],
                                   kwparents={'image': data_sources['image_datasource']}, name='image')

    def intensity_postprocessing_augmented(self, image):
        """
        Intensity postprocessing. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=-128,
                               scale=1/128,
                               random_shift=0.25,
                               random_scale=0.25,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_postprocessing(self, image):
        """
        Intensity postprocessing.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=-128,
                               scale=1/128,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_id_list_file_name,
                                  random=True,
                                  keys=['image_id'])
        data_sources = self.data_sources(True, iterator)
        image_transformation = self.spatial_transformation_augmented(data_sources)
        data_generators = self.data_generators(data_sources, image_transformation, self.intensity_postprocessing_augmented)
        return GraphDataset(data_generators=list(data_generators.values()),
                            data_sources=list(data_sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.val_id_list_file_name,
                                  random=False,
                                  keys=['image_id'])
        data_sources = self.data_sources(False, iterator)
        image_transformation = self.spatial_transformation(data_sources)
        data_generators = self.data_generators(data_sources, image_transformation, self.intensity_postprocessing)
        return GraphDataset(data_generators=list(data_generators.values()),
                            data_sources=list(data_sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)
