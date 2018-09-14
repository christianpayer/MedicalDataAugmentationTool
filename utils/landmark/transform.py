
import SimpleITK as sitk
import numpy as np
import copy
import math
from utils.timer import Timer
import utils.sitk_np
import utils.np_image

def transform_coords(coords, transformation):
    return np.array(transformation.TransformPoint(coords.astype(np.float64)), np.float32)

def transform_landmark(landmark, transformation):
    transformed_landmark = copy.deepcopy(landmark)
    if transformed_landmark.is_valid:
        transformed_landmark.coords = transform_coords(transformed_landmark.coords, transformation)
    return transformed_landmark

def transform_landmarks_inverse(landmarks, transformation, size):
    try:
        inverse = transformation.GetInverse()
        return transform_landmarks(landmarks, inverse)
    except:
        return transform_landmarks_with_resampling(landmarks, transformation, size)

def transform_landmarks(landmarks, transformation):
    transformed_landmarks = []
    for landmark in landmarks:
        transformed_landmarks.append(transform_landmark(landmark, transformation))
    return transformed_landmarks

def transform_landmarks_with_resampling(landmarks, transformation, size, max_min_distance=None):
    transformed_landmarks = copy.deepcopy(landmarks)
    dim = len(size)
    displacement_field = sitk.TransformToDisplacementField(transformation, sitk.sitkVectorFloat32, size=size)
    if dim == 2:
        displacement_field = np.transpose(utils.sitk_np.sitk_to_np(displacement_field), [1, 0, 2])
        mesh = np.meshgrid(np.array(range(size[0]), np.float64),
                           np.array(range(size[1]), np.float64),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=2)

        for i in range(len(transformed_landmarks)):
            if (not transformed_landmarks[i].is_valid) or (transformed_landmarks[i].coords is None):
                continue
            coords = transformed_landmarks[i].coords
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            distances = np.sqrt(vec[:, :, 0] ** 2 + vec[:, :, 1] ** 2)
            invert_min_distance, transformed_coords = utils.np_image.find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            if max_min_distance is not None and min_distance > max_min_distance:
                transformed_landmarks[i].is_valid = False
                transformed_landmarks[i].coords = None
            else:
                transformed_landmarks[i].coords = transformed_coords

    elif dim == 3:
        displacement_field = np.transpose(utils.sitk_np.sitk_to_np(displacement_field), [2, 1, 0, 3])

        mesh = np.meshgrid(np.array(range(size[0]), np.float64),
                           np.array(range(size[1]), np.float64),
                           np.array(range(size[2]), np.float64),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=3)

        for i in range(len(transformed_landmarks)):
            if (not transformed_landmarks[i].is_valid) or (transformed_landmarks[i].coords is None):
                continue
            coords = transformed_landmarks[i].coords
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            distances = np.sqrt(vec[:, :, :, 0] ** 2 + vec[:, :, :, 1] ** 2 + vec[:, :, :, 2] ** 2)
            invert_min_distance, transformed_coords = utils.np_image.find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            if min_distance > max_min_distance:
                transformed_landmarks[i].is_valid = False
                transformed_landmarks[i].coords = None
            else:
                transformed_landmarks[i].coords = transformed_coords
    return transformed_landmarks


