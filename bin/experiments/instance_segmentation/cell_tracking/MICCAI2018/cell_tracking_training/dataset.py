import os
import SimpleITK as sitk
import numpy as np
import math
import random

from datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.label_datasource import LabelDatasource
from generators.image_generator import ImageGenerator
from generators.label_generator import LabelGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, deformation, flip
from utils.np_image import split_label_image, split_label_image_with_unknown_labels, smooth_label_images, merge_label_images, center_of_mass, draw_circle, dilation_circle, distance_transform, roll_with_pad
from transformations.intensity.sitk.smooth import gaussian
from transformations.intensity.sitk.normalize import normalize_robust as normalize_robust_sitk
from utils.io.video_frame_list import VideoFrameList
from generators.video_frame_list_generator import VideoFrameListGenerator
from datasources.video_frame_list_datasource import VideoFrameListDatasource
from utils.random import float_uniform
from transformations.intensity.np.gamma import change_gamma_unnormalized


class Dataset(object):
    """
    The dataset that processes files from the celltracking challenge.
    """
    def __init__(self,
                 image_size,
                 num_frames,
                 base_folder='',
                 train_id_file='tra_train.csv',
                 val_id_file='tra_val.csv',
                 save_debug_images=False,
                 data_format='channels_first',
                 loss_mask_dilation_size=None,
                 loss_weight_dilation_size=3,
                 load_merged=True,
                 load_has_complete_seg=True,
                 load_seg_loss_mask=True,
                 create_instances_merged=True,
                 create_instances_bac=True,
                 seg_mask=True,
                 instance_image_radius_factor=0.2,
                 image_gaussian_blur_sigma=2.0,
                 label_gaussian_blur_sigma=0,
                 image_interpolator='linear',
                 max_num_instances=None,
                 random_jiggle=0.01,
                 random_move=0.05,
                 random_skip_probability=0.0,
                 normalization_consideration_factors=(0.2, 0.1),
                 debug_folder_prefix='',
                 pad_image=True,
                 crop_image_size=None):
        self.image_size = image_size
        self.output_size = image_size
        self.num_frames = num_frames
        self.base_folder = base_folder
        self.image_base_folder = base_folder
        self.setup_base_folder = os.path.join(base_folder, 'setup')
        self.video_frame_list_file_name = os.path.join(self.setup_base_folder, 'frames.csv')
        self.train_id_list_file_name = os.path.join(self.setup_base_folder, train_id_file)
        self.val_id_list_file_name = os.path.join(self.setup_base_folder, val_id_file)
        self.seg_mask_file_name = os.path.join(self.setup_base_folder, 'seg_masks.csv')
        self.save_debug_images = save_debug_images

        self.loss_mask_dilation_size = loss_mask_dilation_size
        self.loss_weight_dilation_size = loss_weight_dilation_size

        self.load_merged = load_merged
        self.load_has_complete_seg = load_has_complete_seg
        self.load_seg_loss_mask = load_seg_loss_mask
        self.create_instances_bac = create_instances_bac
        self.create_instances_merged = create_instances_merged

        self.data_format = data_format
        self.seg_mask = seg_mask
        self.instance_image_radius_factor = instance_image_radius_factor
        self.image_gaussian_blur_sigma = image_gaussian_blur_sigma
        self.label_gaussian_blur_sigma = label_gaussian_blur_sigma
        self.image_interpolator = image_interpolator
        self.pad_image = pad_image
        self.crop_image_size = crop_image_size
        self.max_num_instances = max_num_instances
        self.random_jiggle = random_jiggle
        self.random_move = random_move
        self.random_skip_probability = random_skip_probability
        self.normalization_consideration_factors = normalization_consideration_factors
        self.debug_folder_prefix = debug_folder_prefix

        self.video_frame_stack_axis = 1 if data_format == 'channels_first' else 0
        self.dim = 2

    def get_channel_axis(self, image, data_format):
        """
        Returns the channel axis of the given image.
        :param image: The np array.
        :param data_format: The data format. Either 'channels_first' or 'channels_last'.
        :return: The channel axis.
        """
        if len(image.shape) == 3:
            return 0 if data_format == 'channels_first' else 2
        if len(image.shape) == 4:
            return 0 if data_format == 'channels_first' else 3

    def image_postprocessing(self, image):
        """
        Processes input image files. Blurs, crops, pads, and normalizes according to the parameters.
        :param image: The image np array.
        :return: The processed np array.
        """
        if self.image_gaussian_blur_sigma > 0:
            image = gaussian(image, self.image_gaussian_blur_sigma)
        if self.crop_image_size is not None:
            image = sitk.Crop(image, self.crop_image_size, self.crop_image_size)
        if self.pad_image:
            pad_size = image.GetSize()
            pad_size = [int(s / 2) for s in pad_size]
            image = sitk.MirrorPad(image, pad_size, pad_size)
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        image_float = normalize_robust_sitk(image_float, (-1, 1), consideration_factors=self.normalization_consideration_factors)
        return image_float

    def label_image_postprocessing(self, image):
        """
        Processes label instance segmentation np arrays. Smoothes the image, if self.label_gaussian_blur_sigma > 0.
        :param image: The instance segmentation np array.
        :return: The processed np array.
        """
        if self.label_gaussian_blur_sigma > 0:
            images, labels = split_label_image_with_unknown_labels(image, dtype=np.uint16)
            images = smooth_label_images(images, self.label_gaussian_blur_sigma, dtype=np.uint16)
            image = merge_label_images(images, labels)
        return image

    def datasources(self):
        """
        Returns the data sources that load data for video sequences. Encapsulates entries of datasources_single_frame() into VideoFrameListDatasource.
        :return: A dict of data sources.
        """
        datasources_dict = {}
        datasources_single_frame_dict = self.datasources_single_frame()
        for name, datasource in datasources_single_frame_dict.items():
            datasources_dict[name] = VideoFrameListDatasource(datasource)

        return datasources_dict

    def data_generators(self, image_post_processing):
        """
        Returns the data generators for video sequences. Encapsulates entries of data_generators_single_frame() into VideoFrameListGenerator.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        data_generators_dict = {}
        data_generators_single_frame_dict = self.data_generators_single_frame(None)
        for name, data_generator in data_generators_single_frame_dict.items():
            if name == 'image':
                data_generators_dict[name] = VideoFrameListGenerator(data_generator,
                                                                     stack_axis=self.video_frame_stack_axis,
                                                                     post_processing_np=image_post_processing)
            else:
                data_generators_dict[name] = VideoFrameListGenerator(data_generator,
                                                                     stack_axis=self.video_frame_stack_axis)

        return data_generators_dict

    def datasources_single_frame(self):
        """
        Returns the data sources that load data for a single frame.
        {
        'image:' CachedImageDataSource that loads the image files.
        'merged:' CachedImageDataSource that loads the segmentation/tracking label files.
        'seg_loss_mask:' CachedImageDataSource that loads the mask, which defines, where the region of the image, where the semgentation is valid.
                         This is needed, as the images are not segmented in the border regions of the image.
        'has_complete_seg': LabelDatasource that loads a label file, which contains '1', if the current frame has
                            a complete segmentation (all cells are segmented), or '0' if not.
        }
        :return: A dict of data sources.
        """
        datasources_dict = {}
        # image data source loads input image.
        datasources_dict['image'] = CachedImageDataSource(self.image_base_folder,
                                                          file_ext='.mha',
                                                          set_identity_spacing=True,
                                                          preprocessing=self.image_postprocessing,
                                                          sitk_pixel_type=sitk.sitkUInt16,
                                                          id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '/t' + x['frame_id']})
        if self.load_merged:
            datasources_dict['merged'] = CachedImageDataSource(self.image_base_folder,
                                                               file_ext='.mha',
                                                               set_identity_spacing=True,
                                                               sitk_pixel_type=sitk.sitkUInt16,
                                                               id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '_GT/MERGED/' + x['frame_id']})
        if self.load_seg_loss_mask:
            datasources_dict['seg_loss_mask'] = CachedImageDataSource(self.image_base_folder,
                                                                      file_ext='.mha',
                                                                      set_identity_spacing=True,
                                                                      sitk_pixel_type=sitk.sitkInt8,
                                                                      return_none_if_not_found=False,
                                                                      id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '_GT/seg_loss_mask'})
        if self.load_has_complete_seg:
            datasources_dict['has_complete_seg'] = LabelDatasource(self.seg_mask_file_name,
                                                                   id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '/' + x['frame_id']})
        return datasources_dict

    def data_generators_single_frame(self, image_post_processing):
        """
        Returns the data generators that process a single frame. See datasources_single_frame() for dict values.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        data_generators_dict = {}
        data_generators_dict['image'] = ImageGenerator(self.dim, self.image_size,
                                                       interpolator=self.image_interpolator,
                                                       post_processing_np=image_post_processing,
                                                       data_format=self.data_format,
                                                       resample_default_pixel_value=-1)
        if self.load_merged:
            data_generators_dict['merged'] = ImageGenerator(self.dim, self.output_size,
                                                            interpolator='nearest',
                                                            post_processing_np=self.label_image_postprocessing,
                                                            data_format=self.data_format,
                                                            np_pixel_type=np.uint16)
        if self.load_seg_loss_mask:
            data_generators_dict['seg_loss_mask'] = ImageGenerator(self.dim, self.output_size,
                                                                   interpolator='nearest',
                                                                   data_format=self.data_format,
                                                                   np_pixel_type=np.uint8)
        if self.load_has_complete_seg:
            data_generators_dict['has_complete_seg'] = LabelGenerator()
        return data_generators_dict

    def data_generator_sources(self):
        """
        Returns a dict that defines the connection between datasources and datagenerator parameters for their get() function.
        :return: A dict.
        """
        return {'image': {'image': 'image'},
                'merged': {'image': 'merged'},
                'seg_loss_mask': {'image': 'seg_loss_mask'},
                'has_complete_seg': {'label': 'has_complete_seg'}}

    def binary_labels(self, image):
        """
        Converts an instance label image into a binary label. All instances will be set to 1.
        :param image: The instance label image.
        :return: A list of np arrays. First, is the background. Second is the foreground.
        """
        all_seg = ShiftScaleClamp(clamp_min=0, clamp_max=1)(image)
        return split_label_image(all_seg, [0, 1])

    def loss_mask(self, seg_loss_mask, has_complete_seg):
        """
        Combines the seg_loss_mask (region, where segmentation is valid), with has_complete_set (frames, where segmentation is valid)
        :param seg_loss_mask: The image (np array), where the segmentation is valid.
        :param has_complete_seg: The frames (np array), where the segmentation is valid.
        :return: The image (np array), where the segmentation is valid (1 valid, 0 otherwise).
        """
        return seg_loss_mask * np.reshape(has_complete_seg, has_complete_seg.shape + (1, 1))

    def other_instance_image(self, target_label, other_labels):
        """
        Returns the mask, of neighboring other_labels for a given target_label.
        :param target_label: Image, where pixels of the target label are set to 1, all other pixels are 0.
        :param other_labels: List of images of all labels. Pixels of the label are set to 1, all other pixels are 0.
        :return: The image (np array), where pixels of neighboring labels are set to 2, all other pixels are 0.
        """
        channel_axis = self.get_channel_axis(target_label, self.data_format)
        dim = len(target_label.shape)

        mask = np.zeros_like(target_label)
        circle_radius = self.image_size[0] * self.instance_image_radius_factor
        # handle images with dim 3 (videos) differently
        if dim == 3:
            com = center_of_mass(target_label)
            com = com[1:]
            for i in range(target_label.shape[channel_axis]):
                current_slice = [slice(None), slice(None)]
                current_slice.insert(channel_axis, slice(i, i + 1))
                current_mask = np.squeeze(mask[tuple(current_slice)], axis=channel_axis)
                draw_circle(current_mask, com, circle_radius)
        else:
            com = center_of_mass(target_label)
            draw_circle(mask, com, circle_radius)
        other_instances = np.zeros_like(mask)
        for other_label in other_labels:
            if np.any(np.bitwise_and(mask == 1, other_label == 1)):
                other_instances += other_label
        other_instances[other_instances > 0] = 2

        return other_instances

    def instance_image(self, image):
        """
        Returns the stacked instance images for the current instance segmentation. The resulting np array contains
        images for instances that are stacked at the channel axis. Each entry of the channel axis corresponds to
        a certain instance, where pixels with value 1 indicate the instance, 2 other neighboring instances, and 0 background.
        :param image: The groundtruth instance segmentation.
        :return: The stacked instance image.
        """
        channel_axis = self.get_channel_axis(image, self.data_format)
        image_squeezed = np.squeeze(image)
        labels, _ = split_label_image_with_unknown_labels(image_squeezed, np.uint8)
        if len(labels) == 1:
            return np.zeros_like(image)
        del labels[0]

        if self.max_num_instances is not None and len(labels) > self.max_num_instances:
            random.shuffle(labels)

        instance_images = []
        num_instances = 0
        for label in labels:
            instance_image = self.other_instance_image(target_label=label, other_labels=labels)
            instance_image[label == 1] = 1
            instance_images.append(instance_image)
            num_instances += 1
            if self.max_num_instances is not None and num_instances >= self.max_num_instances:
                break
        return np.stack(instance_images, channel_axis)

    def all_generators_post_processing(self, generators_dict, random_move=False):
        """
        Function that will be called, after all data generators generated their data. Used to combine the results
        of the individual datagenerators. This function will create a dict as follows:
        {
        'image': generators_dict['image'],
        'instances_bac': Instance image for the background. Combines generators_dict['merged'],
                         generators_dict['seg_loss_mask'], and generators_dict['has_complete_seg'].
        'instances_merged': generators_dict['merged']
        :param generators_dict: The generators_dict that will be generated from the dataset.
        :param random_move: If true, the resulting images will be moved randomly according to the given parameters.
        :return: The final generators_dict of np arrays.
        """
        final_generators_dict = {}
        final_generators_dict['image'] = generators_dict['image']

        if self.create_instances_merged:
            instances_seg = self.instance_image(generators_dict['merged'])
            final_generators_dict['instances_merged'] = instances_seg

        if self.create_instances_bac:
            binary_seg = self.binary_labels(generators_dict['merged'])
            seg_loss_mask = self.loss_mask(generators_dict['seg_loss_mask'], generators_dict['has_complete_seg'])
            instances_bac = (binary_seg[0] + 2 * binary_seg[1]) * seg_loss_mask
            final_generators_dict['instances_bac'] = instances_bac

        if random_move:
            final_generators_dict = self.all_generators_post_processing_random_np(final_generators_dict)

        return final_generators_dict

    def all_generators_post_processing_random_np(self, generators_dict):
        """
        Randomly augments images in the generators_dict according to the parameters.
        :param generators_dict: The generators_dict of np arrays.
        :return: The generators_dict with randomly moved np arrays.
        """
        image = generators_dict['image']
        frame_axis = self.video_frame_stack_axis
        random_move_x = float_uniform(-self.image_size[0] * self.random_move, self.image_size[0] * self.random_move)
        random_move_y = float_uniform(-self.image_size[1] * self.random_move, self.image_size[1] * self.random_move)
        for i in range(image.shape[frame_axis]):
            displacement_x = (i / image.shape[frame_axis]) * random_move_x
            displacement_y = (i / image.shape[frame_axis]) * random_move_y
            displacement_x += float_uniform(-self.image_size[0] * self.random_jiggle, self.image_size[0] * self.random_jiggle)
            displacement_y += float_uniform(-self.image_size[1] * self.random_jiggle, self.image_size[1] * self.random_jiggle)
            displacement_x = int(displacement_x)
            displacement_y = int(displacement_y)
            if displacement_x == 0 and displacement_y == 0:
                continue
            for key in generators_dict.keys():
                if len(generators_dict[key].shape) == 4:
                    if key == 'image':
                        generators_dict[key][:, i:i+1, :, :] = roll_with_pad(generators_dict[key][:, i:i+1, :, :], [0, 0, displacement_y, displacement_x], mode='reflect')
                    else:
                        generators_dict[key][:, i:i+1, :, :] = roll_with_pad(generators_dict[key][:, i:i+1, :, :], [0, 0, displacement_y, displacement_x], mode='constant')

        return generators_dict

    def postprocessing_random(self, image):
        """
        Performs random augmentations of a grayscale image. Augmentation consists of random gamma correction,
        random intensity shift/scale per video and per frame.
        :param image: The grayscale image to augment.
        :return: The augmented grayscale image.
        """
        random_lambda = float_uniform(0.7, 1.3)
        image = change_gamma_unnormalized(image, random_lambda)
        image = ShiftScaleClamp(random_shift=0.25, random_scale=0.25)(image)
        if len(image.shape) == 4:
            for i in range(image.shape[self.video_frame_stack_axis]):
                current_slice = [slice(None), slice(None)]
                current_slice.insert(self.video_frame_stack_axis, slice(i, i + 1))
                image[tuple(current_slice)] = ShiftScaleClamp(random_shift=0.05, random_scale=0.05)(image[tuple(current_slice)])
        return image

    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.FitFixedAr(self.dim, self.image_size),
                                    translation.Random(self.dim, [self.image_size[0] * 0.2, self.image_size[1] * 0.2]),
                                    flip.Random(self.dim, [0.5, 0.5]),
                                    rotation.Random(self.dim, [math.pi]),
                                    scale.Random(self.dim, [0.25, 0.25]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size),
                                    deformation.Output(self.dim, [8, 8], 10, self.image_size)])

    def spatial_transformation(self):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.Fit(self.dim, self.image_size),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)])

    def dataset_train(self):
        """
        Returns the training dataset for videos. Random augmentation is performed.
        :return: The training dataset.
        """
        full_video_frame_list_image = VideoFrameList(self.video_frame_list_file_name, int(self.num_frames / 2), int(self.num_frames / 2) - 1,
                                                     border_mode='valid', random_start=True, random_skip_probability=self.random_skip_probability)
        iterator_train = IdListIterator(self.train_id_list_file_name, random=True, keys=['video_id', 'frame_id'],
                                        postprocessing=lambda x: full_video_frame_list_image.get_id_dict_list(x['video_id'], x['frame_id']))

        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        generators_train = self.data_generators(self.postprocessing_random)
        image_transformation = self.spatial_transformation_augmented()

        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        dataset_train = ReferenceTransformationDataset(dim=self.dim,
                                                       reference_datasource_keys={'image': image_key},
                                                       reference_transformation=image_transformation,
                                                       datasources=sources,
                                                       data_generators=generators_train,
                                                       data_generator_sources=generator_sources,
                                                       iterator=iterator_train,
                                                       all_generators_post_processing=lambda x: self.all_generators_post_processing(x, True),
                                                       debug_image_folder=os.path.join(self.debug_folder_prefix, 'debug_train') if self.save_debug_images else None,
                                                       use_only_first_reference_datasource_entry=True)

        return dataset_train

    def dataset_train_single_frame(self):
        """
        Returns the training dataset for single frames. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator_train = IdListIterator(self.train_id_list_file_name, random=True, keys=['video_id', 'frame_id'])

        sources = self.datasources_single_frame()
        generator_sources = self.data_generator_sources()
        generators_train = self.data_generators_single_frame(self.postprocessing_random)
        image_transformation = self.spatial_transformation_augmented()

        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        dataset_train = ReferenceTransformationDataset(dim=self.dim,
                                                       reference_datasource_keys={'image': image_key},
                                                       reference_transformation=image_transformation,
                                                       datasources=sources,
                                                       data_generators=generators_train,
                                                       data_generator_sources=generator_sources,
                                                       iterator=iterator_train,
                                                       all_generators_post_processing=lambda x: self.all_generators_post_processing(x, True),
                                                       debug_image_folder=os.path.join(self.debug_folder_prefix, 'debug_train') if self.save_debug_images else None)

        return dataset_train

    def dataset_val(self):
        """
        Returns the validation dataset for videos. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator_val = IdListIterator(self.val_id_list_file_name, random=False, keys=['video_id', 'frame_id'])

        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        generators_val = self.data_generators(None)
        image_transformation = self.spatial_transformation()

        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        dataset_val = ReferenceTransformationDataset(dim=self.dim,
                                                     reference_datasource_keys={'image': image_key},
                                                     reference_transformation=image_transformation,
                                                     datasources=sources,
                                                     data_generators=generators_val,
                                                     data_generator_sources=generator_sources,
                                                     iterator=iterator_val,
                                                     all_generators_post_processing=lambda x: self.all_generators_post_processing(x, False),
                                                     debug_image_folder=os.path.join(self.debug_folder_prefix, 'debug_val') if self.save_debug_images else None,
                                                     use_only_first_reference_datasource_entry=True)

        return dataset_val

    def dataset_val_single_frame(self):
        """
        Returns the validation dataset for single frames. No random augmentation is performed.
        :return: The validation dataset.
        """
        sources = self.datasources_single_frame()
        generator_sources = self.data_generator_sources()
        generators_val = self.data_generators_single_frame(None)
        image_transformation = self.spatial_transformation()

        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        dataset_val = ReferenceTransformationDataset(dim=self.dim,
                                                     reference_datasource_keys={'image': image_key},
                                                     reference_transformation=image_transformation,
                                                     datasources=sources,
                                                     data_generators=generators_val,
                                                     data_generator_sources=generator_sources,
                                                     iterator=None,
                                                     all_generators_post_processing=lambda x: self.all_generators_post_processing(x, False),
                                                     debug_image_folder=os.path.join(self.debug_folder_prefix, 'debug_val') if self.save_debug_images else None)

        return dataset_val
