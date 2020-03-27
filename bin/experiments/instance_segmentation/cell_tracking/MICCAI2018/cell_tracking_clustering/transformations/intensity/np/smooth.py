
import skimage.filters


def gaussian(image, sigma):
    """
    Multidimensional gaussian smoothing.
    :param image: np image
    :param sigma: list of sigmas per dimension, or scalar for equal sigma in each dimension
    :return: smoothed np image
    """
    return skimage.filters.gaussian(image, sigma)
