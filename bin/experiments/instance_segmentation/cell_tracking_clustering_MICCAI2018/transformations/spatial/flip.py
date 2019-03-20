import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase
from utils.random import bool_bernoulli


class FlipTransformBase(SpatialTransformBase):
    def __init__(self, dim):
        super(FlipTransformBase, self).__init__(dim)

    def get_flip_transform(self, flip_axes):
        """
        Get the flip transform without executing it on an image.
        Flipping is implemented as negative affine scaling.

        :param flip_axes: bool, list of bool
            True flips the axis, False does not flip the axis
        :return: a configured AffineTransform transform
        """
        assert isinstance(flip_axes, list), "Flip axes must be given as list of bool."
        assert len(flip_axes) in [2, 3], "Flip axes must be a list of length 2 for 2D, or 3 for 3D."

        for i in range(len(flip_axes)):
            if flip_axes[i] is True:
                flip_axes[i] = -1.0
            else:
                flip_axes[i] = 1.0

        # print(flip_axes)
        t = sitk.AffineTransform(len(flip_axes))
        t.Scale(flip_axes)

        return t


class Fixed(FlipTransformBase):
    def __init__(self, dim, flip_axes):
        super(Fixed, self).__init__(dim)
        self.current_flip_axes = flip_axes

    def get(self, **kwargs):
        return self.get_flip_transform(self.current_flip_axes)


class Random(FlipTransformBase):
    def __init__(self, dim, random_flip_axes_probs):
        super(Random, self).__init__(dim)
        self.random_flip_axes_probs = random_flip_axes_probs

    def get(self, **kwargs):
        """
        Flip the specified axes according to a given flip probability.

        :param input_image: ITK image
            the input image
        :param random_flip_axes_probs: float, or list of floats
            probabilities for randomly flipping the axes, use 0 to turn off flipping for an axis
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
                   **ONLY FOR use_affine=True**
        :return:
        """
        # flip in each dimension given the flip probability
        if len(self.random_flip_axes_probs) == 1:
            self.current_flip_axes = bool(bool_bernoulli(p=self.random_flip_axes_probs[0]))
            self.current_flip_axes = [self.current_flip_axes] * self.dim
        else:
            # scale by individual factor in each dimension
            current_flip_axes = [bool(bool_bernoulli(p=self.random_flip_axes_probs[i]))
                                      for i in range(self.dim)]
        return self.get_flip_transform(current_flip_axes)

