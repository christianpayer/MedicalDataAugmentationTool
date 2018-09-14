
import SimpleITK as sitk
import utils.sitk_np
import transformations.intensity.np.normalize


def robust_min_max(img, consideration_factors=(0.1, 0.1)):
    return transformations.intensity.np.normalize.robust_min_max(utils.sitk_np.sitk_to_np_no_copy(img), consideration_factors)


def min_max(input_image):
    """
    Computes the min and max intensity of an sitk image.
    :param input_image: The sitk image.
    :return: The minimum and maximum as a list
    """
    min_max_filter = sitk.MinimumMaximumImageFilter()
    min_max_filter.Execute(input_image)
    return [min_max_filter.GetMinimum(), min_max_filter.GetMaximum()]


def scale_min_max(img, old_range, new_range):
    shift = -old_range[0] + new_range[0] * (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
    scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    return sitk.ShiftScale(img, shift, scale)


def normalize(img, out_range=(-1, 1)):
    return sitk.RescaleIntensity(img, out_range[0], out_range[1])


def normalize_robust(img, out_range=(-1, 1), consideration_factors=(0.1, 0.1)):
    min_value, max_value = robust_min_max(img, consideration_factors)
    if max_value == min_value:
        # fix to prevent div by zero
        max_value = min_value + 1
    old_range = (min_value, max_value)
    return scale_min_max(img, old_range, out_range)
