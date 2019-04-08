
import SimpleITK as sitk


def create_composite(dim, transformations):
    """
    Creates a composite sitk transform based on a list of sitk transforms.
    :param dim: The dimension of the transformation.
    :param transformations: A list of sitk transforms.
    :return: The composite sitk transform.
    """
    compos = sitk.Transform(dim, sitk.sitkIdentity)
    for transformation in transformations:
        compos.AddTransform(transformation)
    return compos


def flipped_dimensions(transformation, size):
    """
    Heuristically checks for flipped dimensions. Checks for changes in sign for each dimension.
    :param transformation: The sitk transformation.
    :param size: The size to check, usually the image size.
    :return: List of booleans for each dimension, where True indicates a flipped dimension.
    """
    dim = len(size)
    # transform start point
    start = [0.0] * dim
    transformed_start = transformation.TransformPoint(start)
    flipped = [False] * dim
    for i in range(dim):
        # set current end point and transform it
        end = [0.0] * dim
        end[i] = size[i] or 1.0
        transformed_end = transformation.TransformPoint(end)
        # check, if transformed_start and transformed_end changed position
        flipped[i] = transformed_start[i] > transformed_end[i]
    return flipped
