
import numpy as np
from transformations.intensity.np.normalize import scale


def change_gamma_unnormalized(img, l):
    min_value = np.min(img)
    max_value = np.max(img)
    input_range = (min_value, max_value)
    range_0_1 = (0, 1)
    normalized = scale(img, input_range, range_0_1)
    normalized = change_gamma(normalized, l)
    return scale(normalized, range_0_1, input_range)


def change_gamma(img, l):
    return np.power(img, l)
