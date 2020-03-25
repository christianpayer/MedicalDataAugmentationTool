
import numpy as np


class Landmark(object):
    """
    Landmark object that has coordinates, is_valid, a scale and value.
    """
    def __init__(self,
                 coords=None,
                 is_valid=None,
                 scale=1.0,
                 value=None):
        """
        Initializer.
        :param coords: The landmark coordinates.
        :param is_valid: Defines, if the landmark is valid, i.e., has coordinates.
                         If coords is not None and is_valid is None, self.is_valid will be set to True.
        :param scale: The scale of the landmark.
        :param value: The value of the landmark.
        """
        self.coords = coords
        self.is_valid = is_valid
        if self.is_valid is None:
            self.is_valid = self.coords is not None
        self.scale = scale
        self.value = value


def get_mean_coords(landmarks):
    """
    Returns mean coordinates of a landmark list.
    :param landmarks: Landmark list.
    :return: np.array of mean coordinates.
    """
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return np.nanmean(np.stack(valid_coords, axis=0), axis=0)

def get_mean_landmark(landmarks):
    """
    Returns a Landmark object, where the coordinates are the mean coordinates of the
    given landmark list. scale and value are ignored.
    :param landmarks: Landmark list.
    :return: Landmark object with the mean coordinates.
    """
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return Landmark(np.nanmean(np.stack(valid_coords, axis=0), axis=0))

def get_mean_landmark_list(*landmarks):
    """
    Returns a list of mean Landmarks for two or more given lists of landmarks. The given lists
    must have the same length. The mean of corresponding list entries is calculated with get_mean_landmark.
    :param landmarks: Two or more given lists of landmarks.
    :return: List of mean landmarks.
    """
    return [get_mean_landmark(l) for l in zip(*landmarks)]
