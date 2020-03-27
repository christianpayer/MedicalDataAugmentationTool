import SimpleITK as sitk

from transformations.intensity.base import IntensityTransformBase
from utils import random

def min_max_intensity(input_image):
    """
    Compute the min and max intensity of an ITK image.

    :param input_image: ITK image
        the input image
    :return: list
        [minimum, maximum]
    """
    min_max_filter = sitk.MinimumMaximumImageFilter()
    min_max_filter.Execute(input_image)
    return [min_max_filter.GetMinimum(), min_max_filter.GetMaximum()]


def rescale(input_image, shift=None, scale=None):
    """
    Rescales the intensities of an image (first shifts, then scales).

    :param input_image: ITK image
        the input image
    :param shift: float
        the intensity shift
    :param scale: float
        the scaling factor
    :return: rescaled image, or the input image, if no shift or scale was specified
    """
    if scale is None and shift is None:
        return input_image

    shift_scale_filter = sitk.ShiftScaleImageFilter()
    shift_scale_filter.SetShift(float(shift))
    shift_scale_filter.SetScale(float(scale))
    output_image = shift_scale_filter.Execute(input_image)
    return output_image


def clamp(input_image, clamp_min=None, clamp_max=None):
    """
    Clamp the intensities at a minimum and/or maximum value.

    :param input_image: ITK image
        the input image
    :param clamp_min: float
        minimum value to clamp
    :param clamp_max: float
        maximum value to clamp
    :return: clamped image, or the input image, if no clamp value was specified
    """

    # check if clamping is active
    if clamp_min is not None or clamp_max is not None:
        clamp_filter = sitk.ClampImageFilter()

        if clamp_min and clamp_max:
            # clamp both sides
            clamp_filter.SetLowerBound(float(clamp_min))
            clamp_filter.SetUpperBound(float(clamp_max))

        elif clamp_min and not clamp_max:
            # compute maximum from image
            clamp_max = min_max_intensity(input_image)[1]
            clamp_filter.SetLowerBound(float(clamp_min))
            clamp_filter.SetUpperBound(float(clamp_max))

        elif not clamp_min and clamp_max:
            # compute minimum from image
            clamp_min = min_max_intensity(input_image)[0]
            clamp_filter.SetLowerBound(float(clamp_min))
            clamp_filter.SetUpperBound(float(clamp_max))

        output_image = clamp_filter.Execute(input_image)
        # return the output image
        return output_image

    # return the unchanged input image
    else:
        return input_image


class ShiftScaleClamp(IntensityTransformBase):
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
        self.current_random_shift = 0.0
        self.current_random_scale = 1.0

    def __call__(self, input_image):
        """
        Transforms an image by first shifting and scaling, and then optionally clamps the values.

        :param input_image:
        :param shift:
        :param scale:
        :param clamp_min:
        :param clamp_max:
        :return:
        """
        output_image = input_image
        output_image = rescale(output_image, shift=self.shift, scale=self.scale)
        output_image = rescale(output_image, shift=self.current_random_shift, scale=self.current_random_scale)
        output_image = clamp(output_image, clamp_min=self.clamp_min, clamp_max=self.clamp_max)

        return output_image

    def update(self, **kwargs):
        if self.random_shift is not None:
            self.current_random_shift = random.float_uniform(-self.random_shift, self.random_shift)
        if self.random_scale is not None:
            self.current_random_scale = 1 + random.float_uniform(-self.random_scale, self.random_scale)
