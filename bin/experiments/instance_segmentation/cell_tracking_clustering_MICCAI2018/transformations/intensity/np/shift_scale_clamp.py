
from transformations.intensity.base import IntensityTransformBase
import numpy as np
from utils import random

class ShiftScaleClamp(IntensityTransformBase):
    """
    Performs intensity shifting.
    Order of operations:
        image += shift
        image *= scale
        image += random.float_uniform(-random_shift, random_shift)
        image *= 1 + random.float_uniform(-random_scale, random_scale)
        image = np.clip(image, clamp_min, clamp_max)
    """
    def __init__(self,
                 shift=None,
                 scale=None,
                 clamp_min=None,
                 clamp_max=None,
                 random_shift=None,
                 random_scale=None):
        self.shift = shift
        self.scale = scale
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.random_shift = random_shift
        self.random_scale = random_scale

    def __call__(self, input_image):
        """
        Transforms an image by first shifting and scaling, and then optionally clamps the values.
        Order of operations:
            image += shift
            image *= scale
            image += random.float_uniform(-random_shift, random_shift)
            image *= 1 + random.float_uniform(-random_scale, random_scale)
            image = np.clip(image, clamp_min, clamp_max)
        :param input_image: np input image
        :return: np processed image
        """
        output_image = input_image
        if self.shift is not None:
            output_image += self.shift
        if self.scale is not None:
            output_image *= self.scale
        if self.random_shift is not None:
            current_random_shift = random.float_uniform(-self.random_shift, self.random_shift)
            output_image += current_random_shift
        if self.random_scale is not None:
            current_random_scale = 1 + random.float_uniform(-self.random_scale, self.random_scale)
            output_image *= current_random_scale
        if self.clamp_min is not None or self.clamp_max is not None:
            output_image = np.clip(output_image, self.clamp_min, self.clamp_max)

        return output_image

