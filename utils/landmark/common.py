
import numpy as np

class Landmark(object):
    def __init__(self,
                 coords=None,
                 is_valid=None,
                 scale=1.0,
                 value=None):
        self.coords = coords
        self.is_valid = is_valid
        if self.is_valid == None:
            self.is_valid = self.coords is not None
        self.scale = scale
        self.value = value


def get_mean_coords(landmarks):
    valid_coords = [landmark.coords for landmark in landmarks if landmark.is_valid]
    return np.nanmean(np.stack(valid_coords, axis=0), axis=0)
