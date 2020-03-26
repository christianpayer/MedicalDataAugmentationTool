import os
import SimpleITK as sitk
import numpy as np
import math
import random

from graph.node import LambdaNode
from datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from datasets.graph_dataset import GraphDataset
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
from utils.sitk_image import accumulate


class Dataset(object):
    """
    The dataset that processes files from the celltracking challenge.
    """
    def __init__(self,
                 image_size,
                 num_frames,
                 base_folder='',
                 dataset_name='',
                 train_id_file='tra_train.csv',
                 val_id_file='tra_val.csv',
                 save_debug_images=False,
                 data_format='channels_first',
                 loss_mask_dilation_size=5,
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
                 crop_image_size=None,
                 bitwise_instance_image=False,
                 scale_factor=None):
        self.image_size = image_size
        self.output_size = image_size
        self.num_frames = num_frames
        self.base_folder = base_folder
        self.dataset_name = dataset_name
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
        self.bitwise_instance_image = bitwise_instance_image
        self.max_num_instances = max_num_instances
        self.scale_factor = scale_factor or [1.0, 1.0]

        if self.bitwise_instance_image:
            assert self.max_num_instances is not None, 'if bitwise_instance_image == True, max_num_instances must also be set'
            if self.max_num_instances <= 15:
                self.instances_datatype = np.int32
            elif self.max_num_instances <= 31:
                self.instances_datatype = np.int64
            else:
                raise Exception('max_num_instances is too large for bitwise_instance_image == True')
        else:
            self.instances_datatype = np.uint8


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
        :param image: The sitk image.
        :return: The processed sitk image.
        """
        if self.image_gaussian_blur_sigma > 0:
            image = gaussian(image, [self.image_gaussian_blur_sigma, self.image_gaussian_blur_sigma])
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
        channel_axis = self.get_channel_axis(image, self.data_format)
        if self.label_gaussian_blur_sigma > 0.0: # and self.label_gaussian_blur_sigma != 1.0:
            images, labels = split_label_image_with_unknown_labels(np.squeeze(image, axis=channel_axis), dtype=np.uint16)
            gaussians = [0, self.label_gaussian_blur_sigma, self.label_gaussian_blur_sigma] if image.ndim == 4 else [self.label_gaussian_blur_sigma, self.label_gaussian_blur_sigma]
            images = smooth_label_images(images, gaussians, dtype=np.uint16)
            image = np.expand_dims(merge_label_images(images, labels), channel_axis)
        return image

    def datasources(self, iterator):
        """
        Returns the data sources that load data for video sequences. Encapsulates entries of datasources_single_frame() into VideoFrameListDatasource.
        :return: A dict of data sources.
        """
        datasources_dict = {}
        datasources_single_frame_dict = self.datasources_single_frame(iterator)
        for name, datasource in datasources_single_frame_dict.items():
            if name == 'image' or name == 'merged' or name == 'seg_loss_mask':
                datasources_dict[name] = VideoFrameListDatasource(datasource, postprocessing=accumulate, name=name, parents=[iterator])
            else:
                datasources_dict[name] = VideoFrameListDatasource(datasource, name=name, parents=[iterator])

        return datasources_dict

    def data_generators(self, dim, sources, image_transformation, image_post_processing):
        """
        Returns the data generators for video sequences. Encapsulates entries of data_generators_single_frame() into VideoFrameListGenerator.
        :param dim: Image dimension.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        data_generators_dict = {}
        data_generators_single_frame_dict = self.data_generators_single_frame(dim, sources, image_transformation, image_post_processing)
        for name, data_generator in data_generators_single_frame_dict.items():
            if name == 'image' or name == 'merged' or name == 'seg_loss_mask':
                data_generators_dict[name] = data_generator
            else:
                data_generators_dict[name] = VideoFrameListGenerator(data_generator,
                                                                     stack_axis=self.video_frame_stack_axis,
                                                                     name=name,
                                                                     parents=[sources[name]])

        return data_generators_dict

    def datasources_single_frame(self, iterator):
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
                                                          id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '/t' + x['frame_id']},
                                                          cache_maxsize=1024,
                                                          name='image',
                                                          parents=[iterator])
        if self.load_merged:
            datasources_dict['merged'] = CachedImageDataSource(self.image_base_folder,
                                                               file_ext='.mha',
                                                               set_identity_spacing=True,
                                                               sitk_pixel_type=sitk.sitkUInt16,
                                                               id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '_GT/MERGED/' + x['frame_id']},
                                                               cache_maxsize=1024,
                                                               name='merged',
                                                               parents=[iterator])
        if self.load_seg_loss_mask:
            datasources_dict['seg_loss_mask'] = CachedImageDataSource(self.image_base_folder,
                                                                      file_ext='.mha',
                                                                      set_identity_spacing=True,
                                                                      sitk_pixel_type=sitk.sitkInt8,
                                                                      return_none_if_not_found=False,
                                                                      id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '_GT/seg_loss_mask'},
                                                                      cache_maxsize=1024,
                                                                      name='seg_loss_mask',
                                                                      parents=[iterator])
        if self.load_has_complete_seg:
            datasources_dict['has_complete_seg'] = LabelDatasource(self.seg_mask_file_name,
                                                                   id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '/' + x['frame_id']},
                                                                   name='has_complete_seg',
                                                                   parents=[iterator])
        return datasources_dict

    def data_generators_single_frame(self, dim, datasources, image_transformation, image_post_processing):
        """
        Returns the data generators that process a single frame. See datasources_single_frame() for dict values.
        :param dim: Image dimension.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_size = self.image_size if dim == 2 else self.image_size + [self.num_frames]
        data_generators_dict = {}
        data_generators_dict['image'] = ImageGenerator(dim, image_size,
                                                       interpolator=self.image_interpolator,
                                                       post_processing_np=image_post_processing,
                                                       data_format=self.data_format,
                                                       resample_default_pixel_value=-1,
                                                       name='image',
                                                       parents=[datasources['image'], image_transformation])
        if self.load_merged:
            #interpolator = 'label_gaussian' if self.label_gaussian_blur_sigma == 1.0 else 'nearest'
            data_generators_dict['merged'] = ImageGenerator(dim, image_size,
                                                            interpolator='nearest',
                                                            post_processing_np=self.label_image_postprocessing,
                                                            data_format=self.data_format,
                                                            np_pixel_type=np.uint16,
                                                            name='merged',
                                                            parents=[datasources['merged'], image_transformation])
        if self.load_seg_loss_mask:
            data_generators_dict['seg_loss_mask'] = ImageGenerator(dim, image_size,
                                                                   interpolator='nearest',
                                                                   data_format=self.data_format,
                                                                   np_pixel_type=np.uint8,
                                                                   name='seg_loss_mask',
                                                                   parents=[datasources['seg_loss_mask'], image_transformation])
        if self.load_has_complete_seg:
            data_generators_dict['has_complete_seg'] = LabelGenerator(name='has_complete_seg',
                                                                      parents=[datasources['has_complete_seg']])
        return data_generators_dict

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
        frame_axis = 0
        dim = len(target_label.shape)

        mask = np.copy(target_label) #np.zeros_like(target_label)
        if self.loss_mask_dilation_size > 0:
            if dim == 3:
                for i in range(target_label.shape[frame_axis]):
                    current_slice = [slice(None), slice(None)]
                    current_slice.insert(frame_axis, slice(i, i + 1))
                    mask[tuple(current_slice)] = dilation_circle(np.squeeze(mask[tuple(current_slice)]), (self.loss_mask_dilation_size, self.loss_mask_dilation_size))
            else:
                mask = dilation_circle(mask, (self.loss_mask_dilation_size, self.loss_mask_dilation_size))
        circle_radius = self.image_size[0] * self.instance_image_radius_factor / self.scale_factor[0]
        # handle images with dim 3 (videos) differently
        if dim == 3:
            com = center_of_mass(target_label)
            com = com[1:]
            is_in_frame = np.any(np.any(target_label, axis=1), axis=1)
            min_index = np.maximum(np.min(np.where(is_in_frame)) - 1, 0)
            max_index = np.minimum(np.max(np.where(is_in_frame)) + 1, target_label.shape[frame_axis])
            #for i in range(target_label.shape[frame_axis]):
            for i in range(min_index, max_index):
                current_slice = [slice(None), slice(None)]
                current_slice.insert(frame_axis, slice(i, i + 1))
                #com = center_of_mass(np.squeeze(target_label[tuple(current_slice)]))
                current_mask = np.squeeze(mask[tuple(current_slice)], axis=frame_axis)
                draw_circle(current_mask, com, circle_radius)
        else:
            com = center_of_mass(target_label)
            draw_circle(mask, com, circle_radius)
        other_instances = np.zeros_like(mask)
        for other_label in other_labels:
            if np.array_equal(target_label, other_label):
                continue
            if np.any(np.bitwise_and(mask == 1, other_label == 1)):
                other_instances += other_label
        other_instances[other_instances > 0] = 1

        return other_instances

    def instance_image(self, image, instances_datatype):
        """
        Returns the stacked instance images for the current instance segmentation. The resulting np array contains
        images for instances that are stacked at the channel axis. Each entry of the channel axis corresponds to
        a certain instance, where pixels with value 1 indicate the instance, 2 other neighboring instances, and 0 background.
        :param image: The groundtruth instance segmentation.
        :param instances_datatype: The np datatype for the bitwise instance image.
        :return: The stacked instance image.
        """
        channel_axis = self.get_channel_axis(image, self.data_format)
        image_squeezed = np.squeeze(image, axis=channel_axis)
        labels, _ = split_label_image_with_unknown_labels(image_squeezed, np.uint8)
        if len(labels) == 1:
            return np.zeros_like(image)
        del labels[0]

        if self.max_num_instances is not None and len(labels) > self.max_num_instances:
            random.shuffle(labels)

        instance_image_pairs = []
        num_instances = 0
        for label in labels:
            other_instance_image = self.other_instance_image(target_label=label, other_labels=labels)
            instance_image_pairs.append((label, other_instance_image))
            num_instances += 1
            if self.max_num_instances is not None and num_instances >= self.max_num_instances:
                break
        return self.merge_instance_image_pairs(instance_image_pairs, instances_datatype, channel_axis)

    def merge_instance_image_pairs(self, instance_image_pairs, instances_datatype, channel_axis):
        """
        Merges a list of instance_image_pairs either into a single bitwise image, or concatenates them.
        :param instance_image_pairs: A list of instance_image_pairs. First index of a pair is the instance image,
                                     second index is the image of the neighboring instances.
        :param instances_datatype: The np datatype for the bitwise instance image.
        :return: A bitwise instance image, or concatenated instance images as a np array.
        """
        if self.bitwise_instance_image:
            instance_image_bits = np.zeros_like(instance_image_pairs[0][0], dtype=instances_datatype)
            for i, (instance_image, other_instance_image) in enumerate(instance_image_pairs):
                current_instance_bit = 1 << (i * 2)
                current_instance_image_bits = np.zeros_like(instance_image, dtype=instances_datatype)
                current_instance_image_bits[instance_image == 1] = current_instance_bit
                current_other_instance_bit = 1 << (i * 2 + 1)
                current_other_instance_image_bits = np.zeros_like(instance_image, dtype=instances_datatype)
                current_other_instance_image_bits[other_instance_image == 1] = current_other_instance_bit
                instance_image_bits = np.bitwise_or(instance_image_bits, np.bitwise_or(current_instance_image_bits, current_other_instance_image_bits))
            return np.expand_dims(instance_image_bits, axis=channel_axis)
        else:
            instance_images = []
            for instance_image, other_instance_image in instance_image_pairs:
                instance_images.append(instance_image + other_instance_image * 2)
            return np.stack(instance_images, axis=channel_axis)

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
            final_generators_dict['instances_merged'] = LambdaNode(lambda x: self.instance_image(x, self.instances_datatype), name='instances_merged', parents=[generators_dict['merged']])

        if self.create_instances_bac:
            def instances_bac(merged, seg_loss_mask, has_complete_seg):
                binary_seg = self.binary_labels(merged)
                seg_loss_mask = self.loss_mask(seg_loss_mask, has_complete_seg)
                instances_bac = (binary_seg[0] + 2 * binary_seg[1]) * seg_loss_mask
                return instances_bac.astype(np.int8)
            final_generators_dict['instances_bac'] = LambdaNode(instances_bac, name='instances_bac', parents=[generators_dict['merged'], generators_dict['seg_loss_mask'], generators_dict['has_complete_seg']])

        if random_move:
            random_move = LambdaNode(self.get_random_move, name='random_move', parents=[final_generators_dict['image']])
            for key, value in final_generators_dict.items():
                if key == 'image':
                    f = lambda image, random_move: self.move_volume(image, random_move, 'reflect')
                else:
                    f = lambda image, random_move: self.move_volume(image, random_move, 'constant')
                final_generators_dict[key] = LambdaNode(f, parents=[value, random_move], name=key)

        return final_generators_dict

    def get_random_move(self, image):
        """
        Calculates a list of x/y offsets for a given image.
        :param image: The image.
        :return: List of x/y offset tuples.
        """
        random_move_x = float_uniform(-self.image_size[0] * self.random_move, self.image_size[0] * self.random_move)
        random_move_y = float_uniform(-self.image_size[1] * self.random_move, self.image_size[1] * self.random_move)
        frame_axis = self.video_frame_stack_axis
        random_moves = []
        for i in range(image.shape[frame_axis]):
            displacement_x = (i / image.shape[frame_axis]) * random_move_x
            displacement_y = (i / image.shape[frame_axis]) * random_move_y
            displacement_x += float_uniform(-self.image_size[0] * self.random_jiggle, self.image_size[0] * self.random_jiggle)
            displacement_y += float_uniform(-self.image_size[1] * self.random_jiggle, self.image_size[1] * self.random_jiggle)
            random_moves.append((int(displacement_x), int(displacement_y)))
        return random_moves

    def move_volume(self, image, random_move, mode):
        """
        Randomly augments images in the generators_dict according to the parameters.
        :param generators_dict: The generators_dict of np arrays.
        :return: The generators_dict with randomly moved np arrays.
        """
        frame_axis = self.video_frame_stack_axis
        for i in range(image.shape[frame_axis]):
            displacement_x, displacement_y = random_move[i]
            if displacement_x == 0 and displacement_y == 0:
                continue
            image[:, i:i+1, :, :] = roll_with_pad(image[:, i:i+1, :, :], [0, 0, displacement_y, displacement_x], mode=mode)

        return image

    def postprocessing_random(self, image):
        """
        Performs random augmentations of a grayscale image. Augmentation consists of random gamma correction,
        random intensity shift/scale per video and per frame.
        :param image: The grayscale image to augment.
        :return: The augmented grayscale image.
        """
        random_lambda = float_uniform(0.6, 1.4)
        image = change_gamma_unnormalized(image, random_lambda)
        image = ShiftScaleClamp(random_shift=0.65, random_scale=0.65)(image)
        if len(image.shape) == 4:
            for i in range(image.shape[self.video_frame_stack_axis]):
                current_slice = [slice(None), slice(None)]
                current_slice.insert(self.video_frame_stack_axis, slice(i, i + 1))
                image[tuple(current_slice)] = ShiftScaleClamp(random_shift=0.1, random_scale=0.1)(image[tuple(current_slice)])
        return image

    def spatial_transformation_augmented(self, image):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        # bring image to center and fit to AR
        transformations_list = [translation.InputCenterToOrigin(self.dim),
                                scale.FitFixedAr(self.dim, self.image_size)]
        if self.scale_factor[0] == 1.0 and self.scale_factor[1] == 1.0:
            # if no scale_factor, randomly shift by certain value
            transformations_list.append(translation.Random(self.dim, [self.image_size[0] * 0.35, self.image_size[1] * 0.35]))
        else:
            # else, randomly move in imag size
            move_factor = [(1.0 - self.scale_factor[i]) * 0.5 for i in [0, 1]]
            transformations_list.append(translation.Random(self.dim, [self.image_size[0] * move_factor[0], self.image_size[1] * move_factor[1]]))
            transformations_list.append(scale.Fixed(self.dim, self.scale_factor))
        transformations_list.extend([flip.Random(self.dim, [0.5, 0.5]),
                                     rotation.Random(self.dim, [math.pi]),
                                     scale.RandomUniform(self.dim, 0.25),
                                     scale.Random(self.dim, [0.25, 0.25]),
                                     translation.OriginToOutputCenter(self.dim, self.image_size),
                                     deformation.Output(self.dim, [8, 8], 10, self.image_size)])
        comp = composite.Composite(self.dim, transformations_list,
                                   name='image_transformation_comp',
                                   kwparents={'image': image})
        return LambdaNode(lambda comp: sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, size=self.image_size)),
                          name='image_transformation',
                          kwparents={'comp': comp})

    def spatial_transformation_volumetric_augmented(self, image):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        dim = 3
        transformations_list = [translation.InputCenterToOrigin(dim),
                                scale.FitFixedAr(dim, self.image_size + [None], ignore_dim=[2])]
        if self.scale_factor[0] == 1.0 and self.scale_factor[1] == 1.0:
            # if no scale_factor, randomly shift by certain value
            transformations_list.append(translation.Random(dim, [self.image_size[0] * 0.35, self.image_size[1] * 0.35, 0.0]))
        else:
            # else, randomly move in imag size
            move_factor = [(1.0 - self.scale_factor[i]) * 0.5 for i in [0, 1]]
            transformations_list.append(translation.Random(dim, [self.image_size[0] * move_factor[0], self.image_size[1] * move_factor[1], 0]))
            transformations_list.append(scale.Fixed(dim, self.scale_factor + [1.0]))
        transformations_list.extend([flip.Random(dim, [0.5, 0.5, 0.0]),
                                     rotation.Random(dim, [0., 0., math.pi]),
                                     scale.RandomUniform(dim, 0.25, ignore_dim=[2]),
                                     scale.Random(dim, [0.25, 0.25, 0.0]),
                                     translation.OriginToOutputCenter(dim, self.image_size + [self.num_frames]),
                                     deformation.Output(dim, [6, 6, 4], [10, 10, 0], self.image_size + [self.num_frames])])
        comp = composite.Composite(dim, transformations_list,
                                   name='image_transformation_comp',
                                   kwparents={'image': image})
        return LambdaNode(lambda comp: sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, size=self.image_size + [self.num_frames])),
                          name='image_transformation',
                          kwparents={'comp': comp})

    def spatial_transformation(self, image):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.Fit(self.dim, self.image_size),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)],
                                   name='image_transformation',
                                   kwparents={'image': image})

    def spatial_transformation_volumetric(self, image):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        dim = 3
        return composite.Composite(dim,
                                   [translation.InputCenterToOrigin(dim),
                                    scale.Fit(dim, self.image_size + [self.num_frames]),
                                    translation.OriginToOutputCenter(dim, self.image_size + [self.num_frames])],
                                   name='image_transformation',
                                   kwparents={'image': image})

    def dataset_train(self):
        """
        Returns the training dataset for videos. Random augmentation is performed.
        :return: The training dataset.
        """
        dim = 3
        full_video_frame_list_image = VideoFrameList(self.video_frame_list_file_name, self.num_frames - 1, 0,
                                                     border_mode='valid', random_start=True, random_skip_probability=self.random_skip_probability)
        iterator = IdListIterator(self.train_id_list_file_name, random=True, keys=['video_id', 'frame_id'],
                                        postprocessing=lambda x: full_video_frame_list_image.get_id_dict_list(x['video_id'], x['frame_id']))

        sources = self.datasources(iterator)
        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        image_transformation = self.spatial_transformation_volumetric_augmented(sources[image_key])
        generators = self.data_generators(dim, sources, image_transformation, self.postprocessing_random)
        final_generators = self.all_generators_post_processing(generators, False)

        return GraphDataset(data_generators=list(final_generators.values()),
                            # data_sources=list(sources.values()),
                            # transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_train_single_frame(self):
        """
        Returns the training dataset for single frames. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_id_list_file_name, random=True, keys=['video_id', 'frame_id'])

        sources = self.datasources_single_frame(iterator)
        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        image_transformation = self.spatial_transformation_augmented(sources[image_key])
        generators = self.data_generators_single_frame(2, sources, image_transformation, self.postprocessing_random)
        final_generators = self.all_generators_post_processing(generators, False)

        return GraphDataset(data_generators=list(final_generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset for videos. No random augmentation is performed.
        :return: The validation dataset.
        """
        dim = 3
        full_video_frame_list_image = VideoFrameList(self.video_frame_list_file_name, int(self.num_frames / 2), int(self.num_frames / 2) - 1,
                                                     border_mode='valid', random_start=False, random_skip_probability=0.0)
        iterator = IdListIterator(self.val_id_list_file_name, random=False, keys=['video_id', 'frame_id'])
        iterator_postprocessing = LambdaNode(lambda x: full_video_frame_list_image.get_id_dict_list(x['video_id'], x['frame_id']), parents=[iterator])

        sources = self.datasources(iterator_postprocessing)
        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        image_transformation = self.spatial_transformation_volumetric(sources[image_key])
        generators = self.data_generators(dim, sources, image_transformation, None)
        final_generators = self.all_generators_post_processing(generators, False)

        return GraphDataset(data_generators=list(final_generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val_all_frames(self):
        """
        Returns the validation dataset for videos. No random augmentation is performed.
        :return: The validation dataset.
        """
        dim = 3
        iterator = 'image_ids'
        sources = self.datasources(iterator)
        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        image_transformation = self.spatial_transformation_volumetric(sources[image_key])
        generators = self.data_generators(dim, sources, image_transformation, None)
        final_generators = self.all_generators_post_processing(generators, False)

        return GraphDataset(data_generators=list(final_generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val_single_frame(self):
        """
        Returns the validation dataset for single frames. No random augmentation is performed.
        :return: The validation dataset.
        """
        sources = self.datasources_single_frame('id_dict')
        image_key = 'merged' if self.pad_image or self.crop_image_size is not None else 'image'
        image_transformation = self.spatial_transformation(sources[image_key])
        generators = self.data_generators_single_frame(2, sources, image_transformation, None)
        final_generators = self.all_generators_post_processing(generators, False)

        return GraphDataset(data_generators=list(final_generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[image_transformation],
                            iterator='id_dict',
                            debug_image_folder='debug_val' if self.save_debug_images else None)
