import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class RotationTransformBase(SpatialTransformBase):
    """
    Rotation transformation base class.
    """
    @staticmethod
    def get_rotation_transform(dim, angles):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param angles: List of angles for each dimension (in radians).
        :return: The sitk.AffineTransform().
        """
        if not isinstance(angles, list):
            angles = [angles]
        assert isinstance(angles, list), 'Angles parameter must be a list of floats, one for each dimension.'
        assert len(angles) in [1, 3], 'Angles must be a list of length 1 for 2D, or 3 for 3D.'

        t = sitk.AffineTransform(dim)

        if len(angles) == 1:
            # 2D
            t.Rotate(0, 1, angle=angles[0])
        elif len(angles) > 1:
            # 3D
            # rotate about x axis
            t.Rotate(1, 2, angle=angles[0])
            # rotate about y axis
            t.Rotate(0, 2, angle=angles[1])
            # rotate about z axis
            t.Rotate(0, 1, angle=angles[2])

        return t


class Fixed(RotationTransformBase):
    """
    A rotation transformation with fixed angles (in radian).
    """
    def __init__(self, dim, angles, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param angles: List of angles for each dimension or single value for 2D (in radians).
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Fixed, self).__init__(dim, *args, **kwargs)
        self.current_angles = angles

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        return self.get_rotation_transform(self.dim, self.current_angles)


class Random(RotationTransformBase):
    """
    A rotation transformation with random angles (in radian).
    """
    def __init__(self, dim, random_angles, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param random_angles: List of random angles per dimension. Random angle is calculated uniformly within [-random_angles[i], random_angles[i]]
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Random, self).__init__(dim, *args, **kwargs)
        self.random_angles = random_angles

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        if self.dim == 2:
            self.current_angles = [float_uniform(-self.random_angles[0], self.random_angles[0])]
        elif self.dim == 3:
            # rotate by same random angle in each dimension
            if len(self.random_angles) == 1:
                angle = float_uniform(-self.random_angles[0], self.random_angles[0])
                self.current_angles = [angle] * self.dim
            else:
                # rotate by individual angle in each dimension
                self.current_angles = [float_uniform(-self.random_angles[i], self.random_angles[i])
                                       for i in range(self.dim)]
        return self.get_rotation_transform(self.dim, self.current_angles)
