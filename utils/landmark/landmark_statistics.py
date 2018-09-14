
import numpy as np

class LandmarkStatistics(object):
    def __init__(self):
        self.predicted_landmarks = {}
        self.groundtruth_landmarks = {}
        self.spacings = {}
        self.distances = {}

    def set_groundtruth_and_prediction(self, predicted_landmarks, groundtruth_landmarks, spacings, normalization_factor=1.0, normalization_indizes=None):
        self.predicted_landmarks = predicted_landmarks
        self.groundtruth_landmarks = groundtruth_landmarks
        self.spacings = spacings
        for key in predicted_landmarks.keys():
            self.distances[key] = self.get_distances(predicted_landmarks[key], groundtruth_landmarks[key], spacings[key], normalization_factor, normalization_indizes)

    def add_landmarks(self, id, predicted, groundtruth, spacing=None, normalization_factor=1.0, normalization_indizes=None):
        self.predicted_landmarks[id] = predicted
        self.groundtruth_landmarks[id] = groundtruth
        self.spacings[id] = spacing
        self.distances[id] = self.get_distances(predicted, groundtruth, spacing, normalization_factor, normalization_indizes)

    def get_distance(self, predicted, groundtruth, spacing, normalization_factor):
        if not groundtruth.is_valid or not predicted.is_valid:
            return np.nan
        if spacing is not None:
            return normalization_factor * np.linalg.norm((predicted.coords - groundtruth.coords) * spacing)
        else:
            return normalization_factor * np.linalg.norm(predicted.coords - groundtruth.coords)

    def get_distances(self, predicted, groundtruth, spacing, normalization_factor=1.0, normalization_indizes=None):
        if normalization_indizes is not None:
            normalization_distance = self.get_distance(groundtruth[normalization_indizes[0]], groundtruth[normalization_indizes[1]], None, 1.0)
            if np.isnan(normalization_distance):
                return [np.nan] * len(predicted)
            normalization_factor = normalization_factor / normalization_distance
        return [self.get_distance(l0, l1, spacing, normalization_factor) for (l0, l1) in zip(predicted, groundtruth)]

    def get_pe(self):
        pe = {}
        for id, distances in self.distances.items():
            for i in range(len(distances)):
                pe[id + '_' + str(i)] = distances[i]
        return pe

    def get_ipe(self):
        ipe = {}
        for id, distances in self.distances.items():
            ipe[id] = np.nansum(np.array(list(distances))) / len(distances)
        return ipe

    def get_pe_statistics(self):
        pe = np.array(list(self.distances.values()))
        mean = np.nanmean(pe)
        stdev = np.nanstd(pe)
        median = np.nanmedian(pe)
        return mean, stdev, median

    def get_ipe_statistics(self):
        ipe = np.array(list(self.get_ipe().values()))
        mean = np.nanmean(ipe)
        stdev = np.nanstd(ipe)
        median = np.nanmedian(ipe)
        return mean, stdev, median

    def get_num_outliers(self, radii):
        pe = np.array(list(self.distances.values()))
        radii_outliers = []
        for r in radii:
            num_outliers = np.count_nonzero(pe >= r)
            radii_outliers.append(num_outliers)
        return radii_outliers

    def get_correct_id(self, max_distance):
        """
        Calculates the number of correctly identified landmarks (defined in spine localization dataset).
        A predicted landmark is correct, if the closest landmark is the correct groundtruth landmark and the distance is within max_distance
        :param max_distance: max distance that a landmark can be correct
        :return: # correct
        """
        correct = 0
        for key in self.groundtruth_landmarks.keys():
            groundtruth_landmarks = self.groundtruth_landmarks[key]
            predicted_landmarks = self.predicted_landmarks[key]
            spacing = self.spacings[key]
            for groundtruth_landmark, predicted_landmark in zip(groundtruth_landmarks, predicted_landmarks):
                if not groundtruth_landmark.is_valid or not predicted_landmark.is_valid:
                    continue
                all_landmark_distances = [(other, self.get_distance(predicted_landmark, other, spacing, 1.0)) for other in groundtruth_landmarks if other.is_valid]
                if len(all_landmark_distances) == 0:
                    continue
                closest_landmark_distance = min(all_landmark_distances, key=lambda landmark_distance: landmark_distance[1])
                # closest landmark is the groundtruth landmark and the distance is within max_distance
                if closest_landmark_distance[0] == groundtruth_landmark and closest_landmark_distance[1] <= max_distance:
                    correct += 1
        return correct

    def get_num_valid_distances(self):
        distances_array = np.array(list(self.distances.values()))
        num_distances = distances_array.size
        num_invalid_distances = np.count_nonzero(np.isnan(distances_array))
        num_valid_distances = num_distances - num_invalid_distances
        return num_valid_distances, num_distances

    def get_ipe_overview_string(self):
        stat = self.get_ipe_statistics()
        overview_string = 'IPE:\n'
        overview_string += f'mean:   {stat[0]:.2f}\n'
        overview_string += f'std:    {stat[1]:.2f}\n'
        overview_string += f'median: {stat[2]:.2f}\n'
        return overview_string

    def get_pe_overview_string(self):
        num_valid_distances, num_distances = self.get_num_valid_distances()
        stat = self.get_pe_statistics()
        overview_string = f'PE ({num_valid_distances} out of {num_distances} valid):\n'
        overview_string += f'mean:   {stat[0]:.2f}\n'
        overview_string += f'std:    {stat[1]:.2f}\n'
        overview_string += f'median: {stat[2]:.2f}\n'
        return overview_string

    def get_outliers_pe_string(self, radii):
        outliers = self.get_num_outliers(radii)
        num_valid_distances, _ = self.get_num_valid_distances()
        overview_string = ''
        for r, outlier in zip(radii, outliers):
            overview_string += f'#outliers >= {r}: {outlier} ({100 * outlier / num_valid_distances:.2f}%)\n'
        return overview_string

    def get_per_instance_outliers_string(self, radius):
        overview_string = f'individual #outliers >= {radius}\n'
        for image_id, distances in self.distances.items():
            outliers = [(i, distance) for i, distance in enumerate(distances) if distance >= radius]
            if len(outliers) > 0:
                overview_string += f'{image_id} {len(outliers)} outliers: '
                outlier_strings = [f'{i} ({distance:.2f})' for i, distance in outliers]
                overview_string += ', '.join(outlier_strings) + '\n'
        return overview_string

    def get_correct_id_string(self, max_distance):
        correct_id = self.get_correct_id(max_distance)
        num_valid_distances, _ = self.get_num_valid_distances()
        return f'#correct_id <= {max_distance}: {correct_id} ({100 * correct_id / num_valid_distances:.2f}%)\n'

    def get_overview_string(self, outlier_radii=None, per_instance_outliers_radius=None, correct_id_distance=None):
        overview_string = ''
        overview_string += self.get_pe_overview_string()
        overview_string += self.get_ipe_overview_string()
        if outlier_radii is not None:
            overview_string += self.get_outliers_pe_string(outlier_radii)
        if per_instance_outliers_radius is not None:
            overview_string += self.get_per_instance_outliers_string(per_instance_outliers_radius)
        if correct_id_distance is not None:
            overview_string += self.get_correct_id_string(correct_id_distance)
        return overview_string
