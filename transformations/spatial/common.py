
import numpy as np
import SimpleITK as sitk


def get_affine_homogeneous_matrix(dim, transformation):
    """
    Returns a homogeneous matrix for an affine transformation.
    :param dim: The dimension of the transformation.
    :param transformation: The sitk transformation.
    :return: A homogeneous (dim+1)x(dim+1) matrix as an np.array.
    """
    matrix = np.eye(dim + 1)
    matrix[:dim, :dim] = np.array(transformation.GetMatrix()).reshape([dim, dim]).T
    matrix[dim, :dim] = np.array(transformation.GetTranslation())
    return matrix


def get_affine_matrix_and_translation(dim, homogeneous_matrix):
    """
    Returns an affine transformation parameters for a homogeneous matrix.
    :param dim: The dimension of the transformation.
    :param homogeneous_matrix: The homogeneous (dim+1)x(dim+1) matrix as an np.array.
    :return: A tuple of the homogeneous matrix as a list, and the translation parameters as a list.
    """
    matrix = homogeneous_matrix[:dim, :dim].T.reshape(-1).tolist()
    translation = homogeneous_matrix[dim, :dim].tolist()
    return matrix, translation


def create_composite(dim, transformations, merge_affine=False):
    """
    Creates a composite sitk transform based on a list of sitk transforms.
    :param dim: The dimension of the transformation.
    :param transformations: A list of sitk transforms.
    :param merge_affine: If true, merge affine transformations before calculating the composite transformation.
    :return: The composite sitk transform.
    """
    if merge_affine:
        merged_transformations = []
        combined_matrix = None
        for transformation in transformations:
            if isinstance(transformation, sitk.AffineTransform):
                if combined_matrix is None:
                    combined_matrix = np.eye(dim + 1)
                current_matrix = get_affine_homogeneous_matrix(dim, transformation)
                combined_matrix = current_matrix @ combined_matrix
            else:
                if combined_matrix is not None:
                    matrix, translation = get_affine_matrix_and_translation(dim, combined_matrix)
                    combined_affine_transform = sitk.AffineTransform(dim)
                    combined_affine_transform.SetMatrix(matrix)
                    combined_affine_transform.SetTranslation(translation)
                    merged_transformations.append(combined_affine_transform)
                merged_transformations.append(transformation)
                combined_matrix = None
        if combined_matrix is not None:
            matrix, translation = get_affine_matrix_and_translation(dim, combined_matrix)
            combined_affine_transform = sitk.AffineTransform(dim)
            combined_affine_transform.SetMatrix(matrix)
            combined_affine_transform.SetTranslation(translation)
            merged_transformations.append(combined_affine_transform)
        transformations = merged_transformations

    if sitk.Version_MajorVersion() == 1:
        compos = sitk.Transform(dim, sitk.sitkIdentity)
        for transformation in transformations:
            compos.AddTransform(transformation)
    else:
        compos = sitk.CompositeTransform(transformations)
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
