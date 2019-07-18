
import SimpleITK as sitk
import numpy as np
import copy
import utils.sitk_np
import utils.np_image


def transform_coords(coords, transformation):
    """
    Transforms np coords with a given sitk transformation.
    :param coords: The np coords.
    :param transformation: The sitk transformation
    :return: The transformed np coords.
    """
    return np.array(transformation.TransformPoint(coords.astype(np.float64)), np.float32)


def transform_landmark(landmark, transformation):
    """
    Transforms a landmark object with a given sitk transformation.
    :param landmark: The landmark object.
    :param transformation: The sitk transformation.
    :return: The landmark object with transformed coords.
    """
    transformed_landmark = copy.deepcopy(landmark)
    if transformed_landmark.is_valid:
        transformed_landmark.coords = transform_coords(transformed_landmark.coords, transformation)
    return transformed_landmark


def transform_landmarks_inverse(landmarks, transformation, size, spacing):
    """
    Transforms a landmark object with the inverse of a given sitk transformation. If the transformation
    is not invertible, calculates the inverse by resampling from a dispacement field.
    :param landmarks: The landmark objects.
    :param transformation: The sitk transformation.
    :param size: The size of the output image, on which the landmark should exist.
    :param spacing: The spacing of the output image, on which the landmark should exist.
    :return: The landmark object with transformed coords.
    """
    try:
        inverse = transformation.GetInverse()
        transformed_landmarks = transform_landmarks(landmarks, inverse)
        for transformed_landmark in transformed_landmarks:
            if transformed_landmark.is_valid:
                transformed_landmark.coords /= np.array(spacing)
        return transformed_landmarks
    except:
        # consider a distance of 2 pixels as a maximum allowed distance
        # for calculating the inverse with a transformation field
        max_min_distance = np.max(spacing) * 2
        return transform_landmarks_inverse_with_resampling(landmarks, transformation, size, spacing, max_min_distance)


def transform_landmarks(landmarks, transformation):
    """
    Transforms a list of landmarks for a given sitk transformation.
    :param landmarks: List of landmarks.
    :param transformation: The sitk transformation.
    :return: The list of transformed landmarks.
    """
    transformed_landmarks = []
    for landmark in landmarks:
        transformed_landmarks.append(transform_landmark(landmark, transformation))
    return transformed_landmarks


def transform_landmarks_inverse_with_resampling(landmarks, transformation, size, spacing, max_min_distance=None):
    """
    Transforms a list of landmarks by calculating the inverse of a given sitk transformation by resampling from a displacement field.
    :param landmarks: The list of landmark objects.
    :param transformation: The sitk transformation.
    :param size: The size of the output image, on which the landmark should exist.
    :param spacing: The spacing of the output image, on which the landmark should exist.
    :param max_min_distance: The maximum distance of the coordinate calculated by resampling. If the calculated distance is larger than this value, the landmark will be set to being invalid.
    :return: The landmark object with transformed coords.
    """
    transformed_landmarks = copy.deepcopy(landmarks)
    dim = len(size)
    displacement_field = sitk.TransformToDisplacementField(transformation, sitk.sitkVectorFloat32, size=size, outputSpacing=spacing)
    if dim == 2:
        displacement_field = np.transpose(utils.sitk_np.sitk_to_np(displacement_field), [1, 0, 2])
        mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                           np.array(range(size[1]), np.float32),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=2) * np.expand_dims(np.expand_dims(np.array(spacing, np.float32), axis=0), axis=0)

        for i in range(len(transformed_landmarks)):
            if (not transformed_landmarks[i].is_valid) or (transformed_landmarks[i].coords is None):
                continue
            coords = transformed_landmarks[i].coords
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            #distances = np.sqrt(vec[:, :, 0] ** 2 + vec[:, :, 1] ** 2)
            distances = np.linalg.norm(vec, axis=2)
            invert_min_distance, transformed_coords = utils.np_image.find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            if max_min_distance is not None and min_distance > max_min_distance:
                transformed_landmarks[i].is_valid = False
                transformed_landmarks[i].coords = None
            else:
                transformed_landmarks[i].coords = transformed_coords
    elif dim == 3:
        displacement_field = np.transpose(utils.sitk_np.sitk_to_np(displacement_field), [2, 1, 0, 3])

        mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                           np.array(range(size[1]), np.float32),
                           np.array(range(size[2]), np.float32),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=3) * np.expand_dims(np.expand_dims(np.expand_dims(np.array(spacing, np.float32), axis=0), axis=0), axis=0)

        for i in range(len(transformed_landmarks)):
            if (not transformed_landmarks[i].is_valid) or (transformed_landmarks[i].coords is None):
                continue
            coords = transformed_landmarks[i].coords
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            #distances = np.sqrt(vec[:, :, :, 0] ** 2 + vec[:, :, :, 1] ** 2 + vec[:, :, :, 2] ** 2)
            distances = np.linalg.norm(vec, axis=3)
            invert_min_distance, transformed_coords = utils.np_image.find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            if max_min_distance is not None and min_distance > max_min_distance:
                transformed_landmarks[i].is_valid = False
                transformed_landmarks[i].coords = None
            else:
                transformed_landmarks[i].coords = transformed_coords
    return transformed_landmarks
