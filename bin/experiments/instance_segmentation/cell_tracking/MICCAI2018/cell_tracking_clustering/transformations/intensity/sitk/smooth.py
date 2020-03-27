
import SimpleITK as sitk


def gaussian(image, sigma):
    """
    Multidimensional gaussian smoothing.
    :param image: sitk image
    :param sigma: list of sigmas per dimension, or scalar for equal sigma in each dimension
    :return: smoothed sitk image
    """
    return sitk.SmoothingRecursiveGaussian(image, sigma)
