
import SimpleITK as sitk
from utils import random
from transformations.intensity.sitk.normalize import min_max


def shift_scale(input_image, shift=None, scale=None):
    """
    Rescales the intensities of an image (first shifts, then scales).
    :param input_image: The sitk image.
    :param shift: The intensity shift (added) value.
    :param scale: The intensity scale (multiplied) value.
    :return: The rescaled image.
    """
    if scale is None and shift is None:
        return input_image
    output_image = sitk.ShiftScale(input_image, float(shift), float(scale))
    return output_image


def clamp(input_image, clamp_min=None, clamp_max=None):
    """
    Clamp the intensities at a minimum and/or maximum value.
    :param input_image: The sitk image.
    :param clamp_min: The minimum value to clamp.
    :param clamp_max: The maximum value to clamp.
    :return: The clamped sitk image.
    """
    if clamp_min is not None or clamp_max is not None:
        clamp_filter = sitk.ClampImageFilter()
        if clamp_min and clamp_max:
            clamp_filter.SetLowerBound(float(clamp_min))
            clamp_filter.SetUpperBound(float(clamp_max))
        elif clamp_min and not clamp_max:
            clamp_max = min_max(input_image)[1]
            clamp_filter.SetLowerBound(float(clamp_min))
            clamp_filter.SetUpperBound(float(clamp_max))
        elif not clamp_min and clamp_max:
            clamp_min = min_max(input_image)[0]
            clamp_filter.SetLowerBound(float(clamp_min))
            clamp_filter.SetUpperBound(float(clamp_max))
        output_image = clamp_filter.Execute(input_image)
        return output_image
    else:
        return input_image


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
    if shift is not None or scale is not None:
        output_image = shift_scale(output_image, shift=shift, scale=scale)
    if random_shift is not None or random_scale is not None:
        current_random_shift = random.float_uniform(-random_shift, random_shift)
        current_random_scale = 1 + random.float_uniform(-random_scale, random_scale)
        output_image = shift_scale(output_image, shift=current_random_shift, scale=current_random_scale)
    if clamp_min is not None or clamp_max is not None:
        output_image = clamp(output_image, clamp_min=clamp_min, clamp_max=clamp_max)
    return output_image


class ShiftScaleClamp(object):
    """
    Transforms an image by first shifting and scaling, and then optionally clamps the values.
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
        """
        Initializer.
        :param shift: The intensity shift (added) value (image += shift).
        :param scale: The intensity scale (multiplied) value (image *= scale).
        :param clamp_min: The minimum value to clamp (image = np.clip(image, clamp_min, clamp_max)).
        :param clamp_max: The maximum value to clamp (image = np.clip(image, clamp_min, clamp_max)).
        :param random_shift: The random shift (image += random.float_uniform(-random_shift, random_shift)).
        :param random_scale: The additional random scale (image *= 1 + random.float_uniform(-random_scale, random_scale)).
        """
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
        return shift_scale_clamp(input_image,
                                 shift=self.shift,
                                 scale=self.scale,
                                 clamp_min=self.clamp_min,
                                 clamp_max=self.clamp_max,
                                 random_shift=self.random_shift,
                                 random_scale=self.random_scale)
