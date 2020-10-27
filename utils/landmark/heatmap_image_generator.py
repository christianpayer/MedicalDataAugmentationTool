
import math
import numpy as np


class HeatmapImageGenerator(object):
    """
    Generates numpy arrays of Gaussian landmark images for the given parameters.
    :param image_size: Output image size
    :param sigma: Sigma of Gaussian
    :param scale_factor: Every value of the landmark is multiplied with this value
    :param normalize_center: if true, the value on the center is set to scale_factor
                             otherwise, the default gaussian normalization factor is used
    :param size_sigma_factor: the region size for which values are being calculated
    """
    def __init__(self,
                 image_size,
                 sigma,
                 scale_factor,
                 normalize_center=True,
                 size_sigma_factor=5):
        self.image_size = image_size
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.dim = len(image_size)
        self.normalize_center = normalize_center
        self.size_sigma_factor = size_sigma_factor

    def generate_heatmap(self, coords, sigma_scale_factor, dtype=np.float32):
        """
        Generates a numpy array of the landmark image for the specified point and parameters.
        :param coords: numpy coordinates ([x], [x, y] or [x, y, z]) of the point.
        :param sigma_scale_factor: Every value of the gaussian is multiplied by this value.
        :param dtype: The heatmap output type.
        :return: numpy array of the landmark image.
        """
        # landmark holds the image
        heatmap = np.zeros(self.image_size, dtype=dtype)

        # flip point from [x, y, z] to [z, y, x]
        flipped_coords = np.flip(coords, 0)
        region_start = (flipped_coords - self.sigma * self.size_sigma_factor / 2).astype(int)
        region_end = (flipped_coords + self.sigma * self.size_sigma_factor / 2).astype(int)

        region_start = np.maximum(0, region_start).astype(int)
        region_end = np.minimum(self.image_size, region_end).astype(int)

        # return zero landmark, if region is invalid, i.e., landmark is outside of image
        if np.any(region_start >= region_end):
            return heatmap

        region_size = (region_end - region_start).astype(int)

        sigma = self.sigma * sigma_scale_factor
        scale = self.scale_factor

        if not self.normalize_center:
            scale /= math.pow(math.sqrt(2 * math.pi) * sigma, self.dim)

        if self.dim == 1:
            dx = np.meshgrid(range(region_size[0]))
            x_diff = dx + region_start[0] - flipped_coords[0]

            squared_distances = x_diff * x_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0]] = cropped_heatmap[:]

        if self.dim == 2:
            dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
            x_diff = dx + region_start[0] - flipped_coords[0]
            y_diff = dy + region_start[1] - flipped_coords[1]

            squared_distances = x_diff * x_diff + y_diff * y_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0],
                    region_start[1]:region_end[1]] = cropped_heatmap[:, :]

        elif self.dim == 3:
            dy, dx, dz = np.meshgrid(range(region_size[1]), range(region_size[0]), range(region_size[2]))
            x_diff = dx + region_start[0] - flipped_coords[0]
            y_diff = dy + region_start[1] - flipped_coords[1]
            z_diff = dz + region_start[2] - flipped_coords[2]

            squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0],
                    region_start[1]:region_end[1],
                    region_start[2]:region_end[2]] = cropped_heatmap[:, :, :]

        return heatmap

    def generate_heatmaps(self, landmarks, stack_axis, dtype=np.float32):
        """
        Generates a numpy array landmark images for the specified points and parameters.
        :param landmarks: List of points. A point is a dictionary with the following entries:
            'is_valid': bool, determines whether the coordinate is valid or not
            'coords': numpy coordinates ([x], [x, y] or [x, y, z]) of the point.
            'scale': scale factor of the point.
        :param stack_axis: The axis where to stack the np arrays.
        :param dtype: The heatmap output type.
        :return: numpy array of the landmark images.
        """
        heatmap_list = []

        for landmark in landmarks:
            if landmark.is_valid:
                heatmap_list.append(self.generate_heatmap(landmark.coords, landmark.scale, dtype))
            else:
                heatmap_list.append(np.zeros(self.image_size, dtype))

        heatmaps = np.stack(heatmap_list, axis=stack_axis)

        return heatmaps
    
