
import numpy as np


class LandmarkStatistics(object):
    """
    This class is used to calcualte landmark statistics, e.g., mean point error, outliers, etc
    """
    def __init__(self):
        """
        Initializer.
        """
        self.predicted_landmarks = {}
        self.groundtruth_landmarks = {}
        self.spacings = {}
        self.distances = {}

    def set_groundtruth_and_prediction(self, predicted_landmarks, groundtruth_landmarks, spacings, normalization_factor=1.0, normalization_indizes=None):
        """
        Sets groundtruth and prediction landmark dicts.
        :param predicted_landmarks: Dict of predicted Landmark objects.
        :param groundtruth_landmarks: Dict of groundtruth Landmark objects.
        :param spacings: Dict of image spacings.
        :param normalization_factor: Normalization factor used for distance calculation.
        :param normalization_indizes: Normalization indizes used for distance calculation. The distance of these two point indizes is cosidered to be 1.0.
        :return:
        """
        self.predicted_landmarks = predicted_landmarks
        self.groundtruth_landmarks = groundtruth_landmarks
        self.spacings = spacings
        for key in predicted_landmarks.keys():
            self.distances[key] = self.get_distances(predicted_landmarks[key], groundtruth_landmarks[key], spacings[key], normalization_factor, normalization_indizes)

    def add_landmarks(self, image_id, predicted, groundtruth, spacing=None, normalization_factor=1.0, normalization_indizes=None):
        """
        Add landmarks for a given ID.
        :param image_id: The image id.
        :param predicted: The predicted landmarks.
        :param groundtruth: The groundtruth landmarks.
        :param spacing: The image spacing.
        :param normalization_factor: Normalization factor used for distance calculation.
        :param normalization_indizes: Normalization indizes used for distance calculation. The distance of these two point indizes is cosidered to be 1.0.
        :return:
        """
        self.predicted_landmarks[image_id] = predicted
        self.groundtruth_landmarks[image_id] = groundtruth
        self.spacings[image_id] = spacing
        self.distances[image_id] = self.get_distances(predicted, groundtruth, spacing, normalization_factor, normalization_indizes)

    def get_distance(self, l0, l1, spacing, normalization_factor):
        """
        Returns the distance between two landmarks.
        :param l0: Landmark 1.
        :param l1: Landmark 2.
        :param spacing: The image spacing.
        :param normalization_factor: Normalization factor used for distance calculation.
        :return: The distance.
        """
        if not l1.is_valid or not l0.is_valid:
            return np.nan
        if spacing is not None:
            return normalization_factor * np.linalg.norm((l0.coords - l1.coords) * spacing)
        else:
            return normalization_factor * np.linalg.norm(l0.coords - l1.coords)

    def get_distances(self, predicted, groundtruth, spacing, normalization_factor=1.0, normalization_indizes=None):
        """
        Returns a list of distances between the predicted and groundtruth landmarks.
        :param predicted: List of predicted landmarks.
        :param groundtruth: List of groundtruth landmarks.
        :param spacing: Image spacing.
        :param normalization_factor: Normalization factor used for distance calculation.
        :param normalization_indizes: If not None, the distance of these two point indizes is cosidered to be 1.0.
        :return:
        """
        if normalization_indizes is not None:
            normalization_distance = self.get_distance(groundtruth[normalization_indizes[0]], groundtruth[normalization_indizes[1]], None, 1.0)
            if np.isnan(normalization_distance):
                return [np.nan] * len(predicted)
            normalization_factor = normalization_factor / normalization_distance
        return [self.get_distance(l0, l1, spacing, normalization_factor) for (l0, l1) in zip(predicted, groundtruth)]

    def get_pe(self):
        """
        Returns the point error dictionary of all distances.
        :return: The point errors.
        """
        pe = {}
        for image_id, distances_per_image in self.distances.items():
            for i in range(len(distances_per_image)):
                pe[image_id + '_' + str(i)] = distances_per_image[i]
        return pe

    def get_ipe(self):
        """
        Returns the image point error dictionary of all images.
        :return: The image point errors.
        """
        ipe = {}
        for image_id, distances in self.distances.items():
            ipe[image_id] = np.nansum(np.array(list(distances))) / len(distances)
        return ipe

    def get_pe_statistics(self):
        """
        Returns mean, stddev and median point errors,
        :return: mean, stddev and median
        """
        pe = np.array(list(self.distances.values()))
        mean = np.nanmean(pe)
        stdev = np.nanstd(pe)
        median = np.nanmedian(pe)
        return mean, stdev, median

    def get_ipe_statistics(self):
        """
        Returns mean, stddev and median image point errors,
        :return: mean, stddev and median
        """
        ipe = np.array(list(self.get_ipe().values()))
        mean = np.nanmean(ipe)
        stdev = np.nanstd(ipe)
        median = np.nanmedian(ipe)
        return mean, stdev, median

    def get_num_outliers(self, radii, normalize=False):
        """
        Returns number of point error outliers for given radii.
        :param radii: List of radii.
        :param normalize: If true, divide number of outliers with the total number of points.
        :return: List of number of outliers for the given radii.
        """
        pe = np.array(list(self.distances.values()))
        radii_outliers = []
        for r in radii:
            num_outliers = np.count_nonzero(pe >= r)
            if normalize:
                num_valid_distances, _ = self.get_num_valid_distances()
                num_outliers /= num_valid_distances
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
                all_gt_landmarks_and_distances = [(other_gt_landmark, self.get_distance(predicted_landmark, other_gt_landmark, spacing, 1.0))
                                                  for other_gt_landmark in groundtruth_landmarks if other_gt_landmark.is_valid]
                if len(all_gt_landmarks_and_distances) == 0:
                    continue
                closest_gt_landmark, closest_gt_distance = min(all_gt_landmarks_and_distances, key=lambda landmark_distance: landmark_distance[1])
                # closest landmark is the groundtruth landmark and the distance is within max_distance
                if closest_gt_landmark == groundtruth_landmark and closest_gt_distance <= max_distance:
                    correct += 1
        return correct

    # def get_correct_id(self, max_distance):
    #     """
    #     Calculates the number of correctly identified landmarks (defined in spine localization dataset).
    #     A predicted landmark is correct, if the closest landmark is the correct groundtruth landmark and the distance is within max_distance
    #     :param max_distance: max distance that a landmark can be correct
    #     :return: # correct
    #     """
    #     correct = 0
    #     for key in self.groundtruth_landmarks.keys():
    #         groundtruth_landmarks = self.groundtruth_landmarks[key]
    #         predicted_landmarks = self.predicted_landmarks[key]
    #         spacing = self.spacings[key]
    #         for groundtruth_landmark, predicted_landmark in zip(groundtruth_landmarks, predicted_landmarks):
    #             if not groundtruth_landmark.is_valid or not predicted_landmark.is_valid:
    #                 continue
    #             all_pred_landmarks_and_distances = [(other_pred_landmark, self.get_distance(predicted_landmark, other_pred_landmark, spacing, 1.0))
    #                                                 for other_pred_landmark in predicted_landmarks if other_pred_landmark.is_valid]
    #             if len(all_pred_landmarks_and_distances) == 0:
    #                 continue
    #             closest_pred_landmark, closest_pred_distance = min(all_pred_landmarks_and_distances, key=lambda landmark_distance: landmark_distance[1])
    #             # closest landmark is the groundtruth landmark and the distance is within max_distance
    #             if closest_pred_landmark == predicted_landmark and closest_pred_distance <= max_distance:
    #                 correct += 1
    #     return correct

    def get_num_valid_distances(self):
        """
        Returns the number of valid distances (where groundtruth/prediction is not none) and the number of all distances.
        :return: The number of valid distances and the number of distances.
        """
        distances_array = np.array(list(self.distances.values()))
        num_distances = distances_array.size
        num_invalid_distances = np.count_nonzero(np.isnan(distances_array))
        num_valid_distances = num_distances - num_invalid_distances
        return num_valid_distances, num_distances

    def get_ipe_overview_string(self):
        """
        Returns a overview string of the ipe statistics. Mean, stddev and median.
        :return: Overview string.
        """
        stat = self.get_ipe_statistics()
        overview_string = 'IPE:\n'
        overview_string += 'mean:   {:.2f}\n'.format(stat[0])
        overview_string += 'std:    {:.2f}\n'.format(stat[1])
        overview_string += 'median: {:.2f}\n'.format(stat[2])
        return overview_string

    def get_pe_overview_string(self):
        """
        Returns a overview string of the pe statistics. Mean, stddev and median.
        :return: Overview string.
        """
        num_valid_distances, num_distances = self.get_num_valid_distances()
        stat = self.get_pe_statistics()
        overview_string = 'PE ({} out of {} valid):\n'.format(num_valid_distances, num_distances)
        overview_string += 'mean:   {:.2f}\n'.format(stat[0])
        overview_string += 'std:    {:.2f}\n'.format(stat[1])
        overview_string += 'median: {:.2f}\n'.format(stat[2])
        return overview_string

    def get_outliers_pe_string(self, radii):
        """
        Returns a overview string for the number of outliers.
        :param radii: List of radii.
        :return: Overview string.
        """
        outliers = self.get_num_outliers(radii)
        num_valid_distances, _ = self.get_num_valid_distances()
        overview_string = ''
        for r, outlier in zip(radii, outliers):
            overview_string += '#outliers >= {}: {} ({:.2f}%)\n'.format(r, outlier, 100 * outlier / num_valid_distances)
        return overview_string

    def get_per_instance_outliers_string(self, radius):
        """
        Returns an overview string of all image_ids, if they contain at least an outlier > radius.
        :param radius: Outlier radius.
        :return: Overview string.
        """
        overview_string = 'individual #outliers >= {}\n'.format(radius)
        for image_id, distances in self.distances.items():
            outliers = [(i, distance) for i, distance in enumerate(distances) if distance >= radius]
            if len(outliers) > 0:
                overview_string += '{} {} outliers: '.format(image_id, len(outliers))
                outlier_strings = ['{} ({:.2f})'.format(i, distance) for i, distance in outliers]
                overview_string += ', '.join(outlier_strings) + '\n'
        return overview_string

    def get_correct_id_string(self, max_distance):
        """
        Returns the correct id rate string.
        :param max_distance: Distance to check.
        :return: Overview string.
        """
        correct_id = self.get_correct_id(max_distance)
        num_valid_distances, _ = self.get_num_valid_distances()
        return '#correct_id <= {}: {} ({:.2f}%)\n'.format(max_distance, correct_id, 100 * correct_id / num_valid_distances)

    def get_overview_string(self, outlier_radii=None, per_instance_outliers_radius=None, correct_id_distance=None):
        """
        Returns an overall overview string.
        :param outlier_radii: The outlier radii.
        :param per_instance_outliers_radius: The per instance outlier radius (see get_per_instance_outliers_string())
        :param correct_id_distance: The correct id distance (see get_correct_id_string())
        :return: Overview string.
        """
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
