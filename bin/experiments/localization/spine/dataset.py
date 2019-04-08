import os
import SimpleITK as sitk

from datasets.graph_dataset import GraphDataset
from datasources.image_datasource import ImageDataSource
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGeneratorHeatmap, LandmarkGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, deformation
from transformations.intensity.sitk.smooth import gaussian


class Dataset(object):
    """
    The dataset that processes files from the MICCAI2014 spine localization challenge.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 sigma,
                 num_landmarks,
                 base_folder,
                 cv,
                 data_format,
                 save_debug_images,
                 generate_heatmaps=False,
                 generate_landmarks=True,
                 translate_by_random_factor=True,
                 smoothing_sigma=0.5):
        """
        Initializer.
        :param image_size: The output image size.
        :param image_spacing: The output image spacing.
        :param sigma: The heatmap sigma. Only used, if generate_heatmaps == True.
        :param num_landmarks: The number of landmarks.
        :param base_folder: The base folder of the dataset conaining images, landmarks and setup files.
        :param cv: The cross validation index. If cv == 0 or cv == 1 use cv setup. If -1, use challenge dataset.
        :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
        :param save_debug_images: If True, save each generated image into the debug image folder.
        :param generate_heatmaps: If True, generate heatmap images.
        :param generate_landmarks: If True, generate landmark list.
        :param translate_by_random_factor: If True, do not center the whole image into the image_size, but crop
                                           the image and translate it by a random factor in z axis.
        :param smoothing_sigma: The gaussian sigma used for smoothing the input image before resampling.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.sigma = sigma
        self.num_landmarks = num_landmarks
        self.base_folder = base_folder
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.generate_heatmaps = generate_heatmaps
        self.generate_landmarks = generate_landmarks
        self.translate_by_random_factor = translate_by_random_factor
        self.smoothing_sigma = smoothing_sigma
        self.dim = 3
        self.image_base_folder = os.path.join(self.base_folder, 'images')
        self.setup_base_folder = os.path.join(self.base_folder, 'setup')
        self.point_list_file_name = os.path.join(self.setup_base_folder, 'points_world_coordinates.csv')
        if cv == -1:
            self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'train_all.txt')
            self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'test_all.txt')
        else:
            self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'cv', str(cv), 'train.txt')
            self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'cv', str(cv), 'val.txt')

    def data_sources(self, iterator, cached, use_landmarks=True, image_extension='.nii.gz'):
        """
        Returns the data sources that load data.
        {
        'image_datasource:' ImageDataSource that loads the image files.
        'landmarks_datasource:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param iterator: The iterator node object.
        :param cached: If True, use CachedImageDataSource instead of ImageDataSource.
        :param use_landmarks: If True, create a landmarks datasource.
        :param image_extension: The extension of the image files.
        :return: A dict of data sources.
        """
        if self.smoothing_sigma is not None and self.smoothing_sigma > 0:
            preprocessing = lambda x: sitk.Cast(gaussian(x, self.smoothing_sigma), sitk.sitkInt16)
        else:
            preprocessing = None
        if cached:
            image_datasource = CachedImageDataSource(self.image_base_folder, '', '', image_extension, preprocessing=preprocessing, cache_maxsize=25000, name='image_datasource', parents=[iterator])
        else:
            image_datasource = ImageDataSource(self.image_base_folder, '', '', image_extension, preprocessing=preprocessing, name='image_datasource', parents=[iterator])
        data_sources_dict = {}
        data_sources_dict['image_datasource'] = image_datasource
        if use_landmarks:
            landmark_datasource = LandmarkDataSource(self.point_list_file_name, self.num_landmarks, self.dim, name='landmarks_datasource', parents=[iterator])
            data_sources_dict['landmarks_datasource'] = landmark_datasource
        return data_sources_dict

    def data_generators(self, image_datasource, landmarks_datasource, transformation, image_post_processing_np, image_size, use_landmarks=True):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param image_datasource: The image datasource.
        :param landmarks_datasource: The landmarks datasource.
        :param transformation: The transformation.
        :param image_post_processing_np: The np postprocessing function for the image data generator.
        :param image_size: The image size.
        :param use_landmarks: If True, generate heatmaps or landmarks.
        :return: A dict of data generators.
        """
        data_generators_dict = {}
        data_generators_dict['image'] = ImageGenerator(self.dim,
                                                       image_size,
                                                       self.image_spacing,
                                                       post_processing_np=image_post_processing_np,
                                                       interpolator='linear',
                                                       resample_default_pixel_value=-3024,
                                                       data_format=self.data_format,
                                                       name='image',
                                                       parents=[image_datasource, transformation])
        if use_landmarks:
            if self.generate_heatmaps:
                data_generators_dict['heatmaps'] = LandmarkGeneratorHeatmap(self.dim, image_size, self.image_spacing, self.sigma, 1.0, True,
                                                                            name='heatmaps',
                                                                            parents=[landmarks_datasource, transformation])
            if self.generate_landmarks:
                data_generators_dict['landmarks'] = LandmarkGenerator(self.dim, self.image_size, self.image_spacing,
                                                                      name='landmarks',
                                                                      parents=[landmarks_datasource, transformation])
        return data_generators_dict

    def spatial_transformation_augmented(self, image_datasource):
        """
        The spatial image transformation with random augmentation.
        :param image_datasource: The image datasource.
        :return: The transformation.
        """
        if self.translate_by_random_factor:
            return composite.Composite(self.dim,
                                       [deformation.Input(self.dim, [5, 5, 5], 15),
                                        translation.InputCenterToOrigin(self.dim),
                                        translation.Random(self.dim, [20, 20, 20]),
                                        translation.RandomFactorInput(self.dim, [0, 0, 0.5], [0, 0, self.image_spacing[2] * self.image_size[2]]),
                                        rotation.Random(self.dim, [0.2, 0.2, 0.2]),
                                        scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                        translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                        #deformation.Output(self.dim, [5, 5, 5], 15, self.image_size, self.image_spacing)
                                        ],
                                       name='image',
                                       kwparents={'image': image_datasource})
        else:
            return composite.Composite(self.dim,
                                       [deformation.Input(self.dim, [5, 5, 5], 15),
                                        translation.InputCenterToOrigin(self.dim),
                                        translation.Random(self.dim, [20, 20, 20]),
                                        rotation.Random(self.dim, [0.2, 0.2, 0.2]),
                                        scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                        translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                        #deformation.Output(self.dim, [5, 5, 5], 10, self.image_size, self.image_spacing)
                                        ],
                                       name='image',
                                       kwparents={'image': image_datasource}
                                      )

    def spatial_transformation(self, image_datasource, image_size):
        """
        The spatial image transformation without random augmentation.
        :param image_datasource: The image datasource.
        :param image_size: The image size.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing)],
                                   name='image',
                                   kwparents={'image': image_datasource}
                                   )

    def intensity_postprocessing_augmented(self, image):
        """
        Intensity postprocessing. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0,
                               random_shift=0.15,
                               random_scale=0.15)(image)

    def intensity_postprocessing(self, image):
        """
        Intensity postprocessing.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)


    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_id_list_file_name, random=True, keys=['image_id'])
        data_sources = self.data_sources(iterator, True)
        image_transformation = self.spatial_transformation_augmented(data_sources['image_datasource'])
        data_generators = self.data_generators(data_sources['image_datasource'], data_sources['landmarks_datasource'], image_transformation, self.intensity_postprocessing_augmented, self.image_size)
        dataset = GraphDataset(data_sources=list(data_sources.values()),
                               data_generators=list(data_generators.values()),
                               transformations=[image_transformation],
                               iterator=iterator,
                               debug_image_folder='debug_train' if self.save_debug_images else None)
        return dataset

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.val_id_list_file_name, random=False, keys=['image_id'])
        data_sources = self.data_sources(iterator, False)
        if self.translate_by_random_factor:
            image_size = self.image_size[:2] + [None]
        else:
            image_size = self.image_size
        image_transformation = self.spatial_transformation(data_sources['image_datasource'], image_size)
        data_generators = self.data_generators(data_sources['image_datasource'], data_sources['landmarks_datasource'], image_transformation, self.intensity_postprocessing, image_size)
        dataset = GraphDataset(data_sources=list(data_sources.values()),
                               data_generators=list(data_generators.values()),
                               transformations=[image_transformation],
                               iterator=iterator,
                               debug_image_folder='debug_val' if self.save_debug_images else None)
        return dataset
