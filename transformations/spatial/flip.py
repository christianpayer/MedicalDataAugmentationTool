import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import bool_bernoulli


class FlipTransformBase(SpatialTransformBase):
    """
    Flip transformation base class.
    """
    @staticmethod
    def get_flip_transform(dim, flip_axes):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param flip_axes: List of flip indizes for each dimension. A True entry indicates a dimension to flip.
        :return: The sitk.AffineTransform().
        """
        assert len(flip_axes) == dim, 'flip_axes must have length that is equal to dimension.'

        # a flip is implemented by scaling the image axis by -1.0
        scale_factors = [-1.0 if f else 1.0 for f in flip_axes]

        t = sitk.AffineTransform(dim)
        t.Scale(scale_factors)

        return t


class Fixed(FlipTransformBase):
    """
    A flip transformation with fixed flip axes.
    """
    def __init__(self, dim, flip_axes):
        """
        Initializer.
        :param dim: The dimension.
        :param flip_axes: List of flip indizes for each dimension.
        """
        super(Fixed, self).__init__(dim)
        self.current_flip_axes = flip_axes

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        return self.get_flip_transform(self.dim, self.current_flip_axes)


class Random(FlipTransformBase):
    """
    A flip transformation with a random probability.
    """
    def __init__(self, dim, random_flip_axes_probs):
        """
        Initializer.
        :param dim: The dimension.
        :param random_flip_axes_probs: List of flip probabilities for each dimension.
        """
        super(Random, self).__init__(dim)
        self.random_flip_axes_probs = random_flip_axes_probs

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Not used.
        :return: The sitk.AffineTransform().
        """
        if len(self.random_flip_axes_probs) == 1:
            current_flip_axes = bool(bool_bernoulli(p=self.random_flip_axes_probs[0]))
            current_flip_axes = [current_flip_axes] * self.dim
        else:
            # scale by individual factor in each dimension
            current_flip_axes = [bool(bool_bernoulli(p=self.random_flip_axes_probs[i]))
                                 for i in range(self.dim)]
        return self.get_flip_transform(self.dim, current_flip_axes)
