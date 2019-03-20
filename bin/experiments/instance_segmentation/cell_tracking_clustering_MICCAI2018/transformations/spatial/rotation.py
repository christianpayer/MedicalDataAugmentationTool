import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class RotationTransformBase(SpatialTransformBase):
    def __init__(self, dim):
        super(RotationTransformBase, self).__init__(dim)

    def get_rotation_transform(self, angles):
        """
        Prepare the rotation transform without executing it on an image.

        :param angles: float, list of floats
            the rotation angles around each axis in radians
        :return: SimpleITK.AffineTransform
        """
        if not isinstance(angles, list):
            angles = [angles]
        assert isinstance(angles, list), 'Angles parameter must be a list of floats, one for each dimension.'
        assert len(angles) in [1, 3], 'Angles must be a list of length 1 for 2D, or 3 for 3D.'
        dims = 2 if len(angles) == 1 else 3

        # generate the transform
        t = sitk.AffineTransform(dims)

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

        # return the transform
        return t


class Fixed(RotationTransformBase):
    def __init__(self, dim, angles):
        super(Fixed, self).__init__(dim)
        self.current_angles = angles

    def get(self, **kwargs):
        return self.get_rotation_transform(self.current_angles)


class Random(RotationTransformBase):
    def __init__(self, dim, random_angles):
        super(Random, self).__init__(dim)
        self.random_angles = random_angles

    def get(self, **kwargs):
        """
        Apply a random rotation transform to an input image.
        
        :param input_image: ITK image
            the input image
        :param random_angles: float, list of float
            random rotation angle ranges (in radians) for each dimension
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
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
        return self.get_rotation_transform(self.current_angles)
