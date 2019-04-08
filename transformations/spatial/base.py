
import numpy as np
from graph.node import Node


class SpatialTransformBase(Node):
    """
    A generic spatial transform that can be applied to 2D and 3D images.
    """
    def __init__(self, dim, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension of the transformation.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(SpatialTransformBase, self).__init__(*args, **kwargs)
        self.dim = dim

    def get(self, **kwargs):
        """
        Returns the actual sitk transfrom object with the current parameters.
        :param kwargs: Various arguments that may be used by the transformation, e.g., 'image', 'input_size, 'landmarks', etc.
        :return: sitk transform.
        """
        raise NotImplementedError

    @staticmethod
    def get_image_size_spacing_direction_origin(**kwargs):
        """
        Returns a tuple of (input_size, input_spacing) that is defined by the current kwargs.
        :param kwargs: The current image arguments. Either 'image', or 'input_size' and 'input_spacing'
                       'image': sitk image from which the size and spacing will be read.
                       'input_size': Input size tuple.
                       'input_spacing': Input spacing tuple.
                       'input_origin': Input origin tuple.
        :return: (input_size, input_spacing, input_origin) tuple.
        """
        if 'image' in kwargs:
            assert not ('input_size' in kwargs or 'input_spacing' in kwargs), 'specify either image or input_size and input_spacing, but not both'
            input_image = kwargs.get('image')
            input_size = input_image.GetSize()
            input_spacing = input_image.GetSpacing()
            input_direction = input_image.GetDirection()
            input_origin = input_image.GetOrigin()
            return input_size, input_spacing, input_direction, input_origin
        elif 'input_size' in kwargs and 'input_spacing' in kwargs:
            assert 'image' not in kwargs, 'specify either image or input_size and input_spacing, but not both'
            input_size = kwargs.get('input_size')
            input_spacing = kwargs.get('input_spacing')
            dim = len(input_size)
            input_direction = kwargs.get('input_direction', np.eye(dim).flatten().tolist())
            input_origin = kwargs.get('input_origin', np.zeros(dim).tolist())
            return input_size, input_spacing, input_direction, input_origin
        else:
            raise RuntimeError('specify either image or input_size and input_spacing')

    @staticmethod
    def index_to_physical_point(index, origin, spacing, direction):
        """
        Returns a physical point for an image index and given image metadata.
        :param index: The index to transform.
        :param origin: The image origin.
        :param spacing: The image spacing.
        :param direction: The image direction.
        :return: The transformed point.
        """
        dim = len(index)
        physical_point = np.array(origin) + np.matmul(np.matmul(np.array(direction).reshape([dim, dim]), np.diag(spacing)), np.array(index))
        return physical_point.tolist()

    @staticmethod
    def physical_point_to_index(point, origin, spacing, direction):
        """
        Returns an image index for a physical point and given image metadata.
        :param point: The point to transform.
        :param origin: The image origin.
        :param spacing: The image spacing.
        :param direction: The image direction.
        :return: The transformed point.
        """
        dim = len(point)
        index = np.matmul(np.matmul(np.diag(1 / np.array(spacing)), np.array(direction).reshape([dim, dim]).T), (np.array(point) - np.array(origin)))
        return index.tolist()
