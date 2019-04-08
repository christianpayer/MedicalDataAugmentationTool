
import numpy as np
from utils import random


def shift_scale_clamp(input_image,
                      shift=None,
                      scale=None,
                      clamp_min=None,
                      clamp_max=None,
                      random_shift=None,
                      random_scale=None):
    """
    Transforms an image by first shifting and scaling, and then optionally clamps the values.
    Order of operations:
        image += shift
        image *= scale
        image += random.float_uniform(-random_shift, random_shift)
        image *= 1 + random.float_uniform(-random_scale, random_scale)
        image = np.clip(image, clamp_min, clamp_max)
    :param input_image: The sitk image.
    :param shift: The intensity shift (added) value (image += shift).
    :param scale: The intensity scale (multiplied) value (image *= scale).
    :param clamp_min: The minimum value to clamp (image = np.clip(image, clamp_min, clamp_max)).
    :param clamp_max: The maximum value to clamp (image = np.clip(image, clamp_min, clamp_max)).
    :param random_shift: The random shift (image += random.float_uniform(-random_shift, random_shift)).
    :param random_scale: The additional random scale (image *= 1 + random.float_uniform(-random_scale, random_scale)).
    :return: The transformed sitk image.
    """
    output_image = input_image
    if shift is not None:
        output_image += shift
    if scale is not None:
        output_image *= scale
    if random_shift is not None:
        current_random_shift = random.float_uniform(-random_shift, random_shift)
        output_image += current_random_shift
    if random_scale is not None:
        current_random_scale = 1 + random.float_uniform(-random_scale, random_scale)
        output_image *= current_random_scale
    if clamp_min is not None or clamp_max is not None:
        output_image = np.clip(output_image, clamp_min, clamp_max)

    return output_image


class ShiftScaleClamp(object):
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
        return shift_scale_clamp(input_image, self.shift, self.scale, self.clamp_min, self.clamp_max, self.random_shift, self.random_scale)
