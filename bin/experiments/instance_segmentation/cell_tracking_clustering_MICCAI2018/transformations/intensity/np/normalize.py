import numpy as np


def robust_min_max(img, consideration_factors=(0.1, 0.1)):
    # sort flattened image
    img_sort = np.sort(img, axis=None)
    # consider x% values
    min_median_index = int(img.size * consideration_factors[0] * 0.5)
    max_median_index = int(img.size * (1 - consideration_factors[1] * 0.5))
    # return median of highest x% intensity values
    return img_sort[min_median_index], img_sort[max_median_index]


def scale_min_max(img, old_range, new_range):
    shift = -old_range[0] + new_range[0] * (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
    scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    return (img + shift) * scale
    #normalized_img = (img - old_range[0]) / (old_range[1] - old_range[0])
    #return normalized_img * (new_range[1] - new_range[0]) + new_range[0]


def normalize_mr_robust(img, out_range=(-1, 1), consideration_factor=0.1):
    _, max = robust_min_max(img, (0, consideration_factor))
    old_range = (0, max)
    return scale_min_max(img, old_range, out_range)


def normalize(img, out_range=(-1, 1)):
    min_value = np.min(img)
    max_value = np.max(img)
    old_range = (min_value, max_value)
    return scale_min_max(img, old_range, out_range)


def normalize_robust(img, out_range=(-1, 1), consideration_factors=(0.1, 0.1)):
    min_value, max_value = robust_min_max(img, consideration_factors)
    old_range = (min_value, max_value)
    return scale_min_max(img, old_range, out_range)
