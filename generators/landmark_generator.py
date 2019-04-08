
import numpy as np
from utils.landmark.heatmap_image_generator import HeatmapImageGenerator
import utils.landmark.transform
import transformations.spatial.common
import functools
import copy
from generators.transformation_generator_base import TransformationGeneratorBase


class LandmarkGeneratorBase(TransformationGeneratorBase):
    """
    Base class for creating landmark-based outputs.
    """
    def __init__(self,
                 dim,
                 output_size,
                 output_spacing=None,
                 landmark_indizes=None,
                 landmark_flip_pairs=None,
                 data_format='channels_first',
                 *args, **kwargs):
        """
        Initializer
        :param output_size: output image size
        :param landmark_indizes: list of landmark indizes that will be used for generating the output
        :param landmark_flip_pairs: list of landmark index tuples that will be flipped, if the transformation is flipped
        :param data_format: 'channels_first' of 'channels_last'
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(LandmarkGeneratorBase, self).__init__(dim=dim, *args, **kwargs)
        self.output_size = output_size
        self.output_spacing = output_spacing or [1] * dim
        self.landmark_indizes = landmark_indizes
        self.landmark_flip_pairs = landmark_flip_pairs
        self.data_format = data_format
        if data_format == 'channels_first':
            self.stack_axis = 0
        elif data_format == 'channels_last':
            self.stack_axis = self.dim
        else:
            raise Exception('unsupported data format')

    def is_flipped(self, transformation):
        """
        Returns True, if the current transformation is flipped
        :param transformation: transformation to check
        :return: True, if any dimension of the transformation is flipped, False, otherwise
        """
        flipped = transformations.spatial.common.flipped_dimensions(transformation, self.output_size)
        is_flipped = functools.reduce(lambda a, b: a ^ b, flipped, 0)
        return is_flipped

    def flip_landmarks(self, landmarks, flip):
        """
        Returns flipped landmarks according to self.landmark_flip_pairs
        :param landmarks: list of landmarks to flip
        :param flip: if True, landmarks will be flipped
        :return: list of flipped landmarks
        """
        flipped_landmarks = copy.deepcopy(landmarks)
        if flip and self.landmark_flip_pairs is not None:
            for flip_pair in self.landmark_flip_pairs:
                flipped_landmarks[flip_pair[0]], flipped_landmarks[flip_pair[1]] = flipped_landmarks[flip_pair[1]], flipped_landmarks[flip_pair[0]]
        return flipped_landmarks

    def filter_landmarks(self, landmarks):
        """
        Filter landmarks according to self.landmark_indizes
        :param landmarks: list of landmarks to filter
        :return: list of filtered landmarks
        """
        # if self.landmark_indizes is not set, just return full list
        if self.landmark_indizes is None:
            return copy.deepcopy(landmarks)
        # otherwise, filter with self.landmark_indizes
        filtered_landmarks = []
        for landmark_index in self.landmark_indizes:
            filtered_landmarks.append(copy.deepcopy(landmarks[landmark_index]))
        return filtered_landmarks

    def flip_and_filter_landmarks(self, landmarks, flip):
        """
        Flip landmarks and filter afterwards
        :param landmarks: list of landmarks to flip and filter
        :param flip: if True, landmarks will be flipped
        :return: list of flipped and filtered
        """
        return self.filter_landmarks(self.flip_landmarks(landmarks, flip))

    def transform_landmarks(self, landmarks, transformation):
        """
        Transform landmarks according to transformation
        :param landmarks: list of landmarks to transform
        :param transformation: transformation to perform
        :return: list of transformed landmarks
        """
        return utils.landmark.transform.transform_landmarks_inverse(landmarks, transformation, self.output_size, self.output_spacing)

    def preprocess_landmarks(self, landmarks, transformation, flip):
        """
        Flip, filter and transform landmarks
        :param landmarks: list of landmarks to flip, filter and transform
        :param transformation: transformation to perform
        :param flip: if True, landmarks will be flipped
        :return: list of flipped, filtered and transformed landmarks
        """
        return self.transform_landmarks(self.flip_and_filter_landmarks(landmarks, flip), transformation)


class LandmarkGenerator(LandmarkGeneratorBase):
    """
    Generates a numpy array of landmark coordinates. The output shape will be [num_landmarks, dim + 1].
    The first entry in the second dimension defines, whether the landmark is valid.
    """
    def get(self, landmarks, transformation):
        """
        Return generated heatmaps
        :param landmarks: list of landmarks
        :param transformation: transformation to transform landmarks
        :return: landmarks with shape [num_landmarks, dim + 1]
        """
        flip = self.is_flipped(transformation)
        preprocessed_landmarks = self.preprocess_landmarks(landmarks, transformation, flip)
        output = np.zeros((len(preprocessed_landmarks), self.dim + 1), dtype=np.float32)
        for i, preprocessed_landmark in enumerate(preprocessed_landmarks):
            if preprocessed_landmark.is_valid and np.all(preprocessed_landmark.coords >= 0) and np.all(preprocessed_landmark.coords < self.output_size):
                output[i, :] = [1] + list(reversed(preprocessed_landmark.coords.tolist()))

        return output


class LandmarkGeneratorMultiple(LandmarkGeneratorBase):
    """
    Generates a numpy array of multiple landmark coordinates. The output shape will be [num_instances, num_landmarks, dim + 1].
    The first entry in the third dimension defines, whether the landmark is valid.
    """
    def get(self, landmarks_multiple, transformation):
        """
        Return generated heatmaps
        :param landmarks_multiple: list of list of landmarks
        :param transformation: transformation to transform landmarks
        :return: landmarks with shape [num_instances, num_landmarks, dim + 1]
        """
        flip = self.is_flipped(transformation)
        outputs = []
        for landmarks in landmarks_multiple:
            preprocessed_landmarks = self.preprocess_landmarks(landmarks, transformation, flip)
            output = np.zeros((len(preprocessed_landmarks), self.dim + 1), dtype=np.float32)
            for i, preprocessed_landmark in enumerate(preprocessed_landmarks):
                if preprocessed_landmark.is_valid and np.all(preprocessed_landmark.coords >= 0) and np.all(preprocessed_landmark.coords < self.output_size):
                    output[i, :] = [1] + list(reversed(preprocessed_landmark.coords.tolist()))
            outputs.append(output)

        return np.stack(outputs, axis=0)


class LandmarkGeneratorHeatmap(LandmarkGeneratorBase):
    """
    Generates images of Gaussian heatmaps
    """
    def __init__(self,
                 dim,
                 output_size,
                 output_spacing,
                 sigma,
                 scale_factor,
                 normalize_center,
                 *args, **kwargs):
        """
        Initializer
        :param dim: Dimension
        :param output_size: output image size
        :param sigma: Gaussian sigma
        :param scale_factor: heatmap scale factor, each value of the Gaussian will be multiplied with this value
        :param normalize_center: if True, the value on the center is set to scale_factor
                                 otherwise, the default gaussian normalization factor is used
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(LandmarkGeneratorHeatmap, self).__init__(dim=dim, output_size=output_size, output_spacing=output_spacing, *args, **kwargs)
        self.output_size_np = list(reversed(self.output_size))
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.normalize_center = normalize_center

    def get(self, landmarks, transformation):
        """
        Return generated heatmaps
        :param landmarks: list of landmarks
        :param transformation: transformation to transform landmarks
        :return: generated heatmaps
        """
        flip = self.is_flipped(transformation)
        preprocessed_landmarks = self.preprocess_landmarks(landmarks, transformation, flip)
        heatmap_image_generator = HeatmapImageGenerator(image_size=self.output_size_np,
                                                        sigma=self.sigma,
                                                        scale_factor=self.scale_factor,
                                                        normalize_center=self.normalize_center)
        heatmaps = heatmap_image_generator.generate_heatmaps(preprocessed_landmarks, self.stack_axis)
        return heatmaps


class LandmarkGeneratorMultipleHeatmap(LandmarkGeneratorBase):
    """
    Generates heatmap images with multiple Gaussian peaks
    """
    def __init__(self,
                 dim,
                 output_size,
                 sigma,
                 scale_factor,
                 normalize_center,
                 *args, **kwargs):
        """
        Initializer
        :param dim: Dimension
        :param output_size: output image size
        :param sigma: Gaussian sigma
        :param scale_factor: heatmap scale factor, each value of the Gaussian will be multiplied with this value
        :param normalize_center: if True, the value on the center is set to scale_factor
                                 otherwise, the default gaussian normalization factor is used
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(LandmarkGeneratorMultipleHeatmap, self).__init__(dim=dim, output_size=output_size, *args, **kwargs)
        self.output_size_np = list(reversed(self.output_size))
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.normalize_center = normalize_center

    def get(self, landmarks_multiple, transformation):
        """
        Return generated heatmaps
        :param landmarks_multiple: list of list of landmarks
        :param transformation: transformation to transform landmarks
        :return: generated heatmaps
        """
        flip = self.is_flipped(transformation)
        heatmaps = None
        for i, landmarks in enumerate(landmarks_multiple):
            preprocessed_landmarks = self.preprocess_landmarks(landmarks, transformation, flip)
            heatmap_image_generator = HeatmapImageGenerator(image_size=self.output_size_np,
                                                            sigma=self.sigma,
                                                            scale_factor=self.scale_factor,
                                                            normalize_center=self.normalize_center)
            current_heatmaps = heatmap_image_generator.generate_heatmaps(preprocessed_landmarks, self.stack_axis)
            if heatmaps is None:
                heatmaps = current_heatmaps
            else:
                heatmaps = np.maximum(heatmaps, current_heatmaps)
        return heatmaps


class LandmarkGeneratorMask(LandmarkGeneratorBase):
    """
    Generates images filled with 1 for valid landmarks, and 0 for invalid landmarks
    """
    def __init__(self,
                 dim,
                 output_size,
                 ones_if_every_point_is_invalid=False,
                 *args, **kwargs):
        """
        Initializer
        :param dim: Dimension
        :param output_size: output image size
        :param ones_if_every_point_is_invalid: if True, create ones mask, if every point is invalid
                                               otherwise, create zeros mask, if every point is invalid
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(LandmarkGeneratorMask, self).__init__(dim, output_size, *args, **kwargs)
        self.output_size_np = list(reversed(self.output_size))
        self.ones_if_every_point_is_invalid = ones_if_every_point_is_invalid

    def get(self, landmarks, transformation):
        """
        Return generated heatmaps
        :param landmarks: list of landmarks
        :param transformation: transformation to transform landmarks
        :return: generated heatmaps
        """
        flip = self.is_flipped(transformation)
        # although, we do not need the landmark's coordinate, we still need to preprocess the landmarks
        # to determine which landmark is_valid
        preprocessed_landmarks = self.flip_and_filter_landmarks(landmarks, flip)

        if self.ones_if_every_point_is_invalid:
            # if there is no point valid -> create ones mask, as the person is not visible on the frame
            # useful / needed for tracking
            if all([not landmark.is_valid for landmark in preprocessed_landmarks]):
                current_output_size_np = np.insert(np.array(self.output_size_np), self.stack_axis, np.array([len(preprocessed_landmarks)]))
                return np.ones(current_output_size_np, np.float32)

        # append ones or zeros depending on landmark.is_valid
        mask_list = []
        for landmark in preprocessed_landmarks:
            if landmark.is_valid:
                mask_list.append(np.ones(self.output_size_np, np.float32))
            else:
                mask_list.append(np.zeros(self.output_size_np, np.float32))
        mask = np.stack(mask_list, axis=self.stack_axis)
        return mask
