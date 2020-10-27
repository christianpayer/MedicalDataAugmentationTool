
import numpy as np


class LandmarkStatistics(object):
    """
    This class is used to calculate landmark statistics, e.g., mean point error, outliers, etc
    """
    def __init__(self, float_format_string=':.3f', missed_landmark_pe=np.nan):
        """
        Initializer.
        """
        self.float_format_string = float_format_string
        self.missed_landmark_pe = missed_landmark_pe
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
        :param normalization_indizes: Normalization indizes used for distance calculation. The distance of these two point indizes is considered to be 1.0.
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

    def get_distance(self, prediction, groundtruth, spacing, normalization_factor):
        """
        Returns the distance between two landmarks.
        :param prediction: Predicted landmark.
        :param groundtruth: Groundtruth landmark.
        :param spacing: The image spacing.
        :param normalization_factor: Normalization factor used for distance calculation.
        :return: The distance.
        """
        if groundtruth.is_valid and not prediction.is_valid:
            # landmark is in groundtruth, but not in prediction -> missed
            return self.missed_landmark_pe
        elif not groundtruth.is_valid or not prediction.is_valid:
            # landmark is not in groundtruth or it is not in prediction -> the distance cannot be calculated
            return np.nan
        if spacing is not None:
            return normalization_factor * np.linalg.norm((groundtruth.coords - prediction.coords) * spacing)
        else:
            return normalization_factor * np.linalg.norm(groundtruth.coords - prediction.coords)

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

    def get_pe_statistics(self, landmark_indizes=None):
        """
        Returns mean, stddev and median point errors,
        :return: mean, stddev and median
        """
        if landmark_indizes is None:
            pe = np.array(list(self.distances.values()))
        else:
            pe = np.array(list(self.distances.values()))[:, landmark_indizes]
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

    def get_num_outliers(self, radii, normalize=False, landmark_indizes=None):
        """
        Returns number of point error outliers for given radii.
        :param radii: List of radii.
        :param normalize: If true, divide number of outliers with the total number of points.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: List of number of outliers for the given radii.
        """
        if landmark_indizes is None:
            pe = np.array(list(self.distances.values()))
        else:
            pe = np.array(list(self.distances.values()))[:, landmark_indizes]
        radii_outliers = []
        for r in radii:
            num_outliers = np.count_nonzero(pe >= r)
            if normalize and num_outliers > 0:
                num_valid_distances, _, _ = self.get_num_valid()
                if num_valid_distances > 0:
                    num_outliers /= num_valid_distances
                else:
                    num_outliers = 0
            radii_outliers.append(num_outliers)
        return radii_outliers

    def get_num_correct_id(self, max_distance, landmark_indizes=None):
        """
        Calculates the number of correctly identified landmarks (defined in spine localization dataset).
        A predicted landmark is correct, if the closest landmark is the correct groundtruth landmark and the distance is within max_distance
        :param max_distance: max distance that a landmark can be correct
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: # correct
        """
        num_correct = 0
        for image_id in self.groundtruth_landmarks.keys():
            correct = self.get_correct_id_per_instance(image_id, max_distance, landmark_indizes)
            num_correct += len(correct)
        return num_correct

    def get_correct_id_per_instance(self, image_id, max_distance, landmark_indizes=None):
        """
        Calculates the number of correctly identified landmarks for an image_id (defined in spine localization dataset).
        A predicted landmark is correct, if the closest landmark is the correct groundtruth landmark and the distance is within max_distance
        :param image_id: The image_id for which to calculate the # correct id.
        :param max_distance: max distance that a landmark can be correct
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: # correct
        """
        correct = []
        groundtruth_landmarks = self.groundtruth_landmarks[image_id]
        predicted_landmarks = self.predicted_landmarks[image_id]
        other_groundtruth_landmarks = self.groundtruth_landmarks[image_id]
        if landmark_indizes is not None:
            groundtruth_landmarks = [groundtruth_landmarks[i] for i in landmark_indizes]
            predicted_landmarks = [predicted_landmarks[i] for i in landmark_indizes]
        spacing = self.spacings[image_id]
        for i, (groundtruth_landmark, predicted_landmark) in enumerate(zip(groundtruth_landmarks, predicted_landmarks)):
            if not groundtruth_landmark.is_valid or not predicted_landmark.is_valid:
                continue
            all_groundtruth_landmarks_and_distances = [(other_groundtruth_landmark, self.get_distance(predicted_landmark, other_groundtruth_landmark, spacing, 1.0))
                                              for other_groundtruth_landmark in other_groundtruth_landmarks if other_groundtruth_landmark.is_valid]
            if len(all_groundtruth_landmarks_and_distances) == 0:
                continue
            closest_groundtruth_landmark, closest_groundtruth_distance = min(all_groundtruth_landmarks_and_distances, key=lambda landmark_distance: landmark_distance[1])
            # closest landmark is the groundtruth landmark and the distance is within max_distance
            if closest_groundtruth_landmark == groundtruth_landmark and closest_groundtruth_distance <= max_distance:
                correct.append(i)
        return correct

    def get_num_valid(self, landmark_indizes=None):
        """
        Returns the number of valid distances (where groundtruth/prediction is not none) and the number of all distances.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: The number of valid distances and the number of distances.
        """
        num_valid_distances = 0
        num_valid_groundtruth = 0
        num_valid_predicted = 0
        for image_id in self.groundtruth_landmarks.keys():
            valid_distances, valid_groundtruth, valid_predicted = self.get_valid_per_instance(image_id, landmark_indizes)
            num_valid_distances += len(valid_distances)
            num_valid_groundtruth += len(valid_groundtruth)
            num_valid_predicted += len(valid_predicted)
        return num_valid_distances, num_valid_groundtruth, num_valid_predicted

    def get_valid_per_instance(self, image_id, landmark_indizes=None):
        """
        Returns the number of valid distances for an image_id (where groundtruth/prediction is not none) and the number of all distances.
        :param image_id: The image_id for which to calculate the # correct id.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: The number of valid distances and the number of distances.
        """
        valid_distances = []
        valid_groundtruth = []
        valid_predicted = []
        groundtruth_landmarks = self.groundtruth_landmarks[image_id]
        predicted_landmarks = self.predicted_landmarks[image_id]
        if landmark_indizes is not None:
            groundtruth_landmarks = [groundtruth_landmarks[i] for i in landmark_indizes]
            predicted_landmarks = [predicted_landmarks[i] for i in landmark_indizes]
        for i, (gt, pred) in enumerate(zip(groundtruth_landmarks, predicted_landmarks)):
            if gt.is_valid and pred.is_valid:
                valid_distances.append(i)
            if pred.is_valid:
                valid_predicted.append(i)
            if gt.is_valid:
                valid_groundtruth.append(i)
        return valid_distances, valid_groundtruth, valid_predicted

    def get_num_missed_landmarks(self, landmark_indizes=None):
        """
        Return number of missed and too many predicted landmarks for all images and landmark_indizes.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return num_missed, num_too_many
        """
        num_missed = 0
        num_too_many = 0
        for image_id in self.groundtruth_landmarks.keys():
            missed, too_many = self.get_missed_landmarks_per_instance(image_id, landmark_indizes)
            num_missed += len(missed)
            num_too_many += len(too_many)
        return num_missed, num_too_many

    def get_missed_landmarks_per_instance(self, image_id, landmark_indizes=None):
        """
        Return the list of missed and too many predicted landmarks for a given image_id and landmark_indizes.
        :param image_id: The image_id for which to calculate the list of missed and too many predicted landmarks.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return list of missed, list of too_many
        """
        missed = []
        too_many = []
        groundtruth_landmarks = self.groundtruth_landmarks[image_id]
        predicted_landmarks = self.predicted_landmarks[image_id]
        if landmark_indizes is not None:
            groundtruth_landmarks = [groundtruth_landmarks[i] for i in landmark_indizes]
            predicted_landmarks = [predicted_landmarks[i] for i in landmark_indizes]
        for i, (pred, gt) in enumerate(zip(predicted_landmarks, groundtruth_landmarks)):
            if not pred.is_valid and gt.is_valid:
                missed.append(i)
            if pred.is_valid and not gt.is_valid:
                too_many.append(i)
        return missed, too_many

    def get_ipe_overview_string(self):
        """
        Returns a overview string of the ipe statistics. Mean, stddev and median.
        :return: Overview string.
        """
        stat = self.get_ipe_statistics()
        overview_string = 'IPE:\n'
        overview_string += ('mean:   {' + self.float_format_string + '}\n').format(stat[0])
        overview_string += ('std:    {' + self.float_format_string + '}\n').format(stat[1])
        overview_string += ('median: {' + self.float_format_string + '}\n').format(stat[2])
        return overview_string

    def get_pe_overview_string(self, landmark_indizes=None):
        """
        Returns a overview string of the pe statistics. Mean, stddev and median.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: Overview string.
        """
        num_valid_distances, num_valid_gt, num_valid_pred = self.get_num_valid(landmark_indizes)
        stat = self.get_pe_statistics(landmark_indizes)
        overview_string = 'PE ({} out of {} valid gt; {} valid pred):\n'.format(num_valid_distances, num_valid_gt, num_valid_pred)
        overview_string += ('mean:   {' + self.float_format_string + '}\n').format(stat[0])
        overview_string += ('std:    {' + self.float_format_string + '}\n').format(stat[1])
        overview_string += ('median: {' + self.float_format_string + '}\n').format(stat[2])
        return overview_string

    def get_outliers_pe_string(self, radii):
        """
        Returns a overview string for the number of outliers.
        :param radii: List of radii.
        :return: Overview string.
        """
        outliers = self.get_num_outliers(radii)
        num_valid_distances, _, _ = self.get_num_valid()
        overview_string = ''
        for r, outlier in zip(radii, outliers):
            overview_string += ('#outliers >= {}: {} ({' + self.float_format_string + '}%)\n').format(r, outlier, 100 * outlier / num_valid_distances if num_valid_distances > 0 else 0)
        return overview_string

    def get_per_instance_string(self, outlier_radius, correct_id_max_distance):
        """
        Returns an overview string of all image_ids, if they contain at least an outlier > radius.
        :param outlier_radius: Outlier radius.
        :param correct_id_max_distance: correct_id max distance.
        :return: Overview string.
        """
        overview_string = 'individual #outliers >= {}\n'.format(outlier_radius)
        for image_id in self.groundtruth_landmarks.keys():
            distances = self.distances[image_id]
            outliers = [(i, distance) for i, distance in enumerate(distances) if distance >= outlier_radius]
            missed, too_many = self.get_missed_landmarks_per_instance(image_id)
            correct_id = self.get_correct_id_per_instance(image_id, correct_id_max_distance)
            _, valid_gt, _ = self.get_valid_per_instance(image_id)
            if len(outliers) > 0 or len(missed) > 0 or (len(correct_id) != len(valid_gt)):
                overview_string += f'{image_id}: #miss: {len(missed): >2}, #o_r{outlier_radius}: {len(outliers): >2}, #not_id_r{correct_id_max_distance}: {len(valid_gt) - len(correct_id)}\n'
                if len(outliers) > 0:
                    outlier_strings = [('{} ({' + self.float_format_string + '})').format(i, distance) for i, distance in outliers]
                    overview_string += f'outliers > {outlier_radius}: ' + ', '.join(outlier_strings) + '\n'
                if len(valid_gt) - len(correct_id) > 0:
                    not_id_strings = [f'{i}' for i in valid_gt if i not in correct_id]
                    overview_string += f'not_id > {correct_id_max_distance}: ' + ', '.join(not_id_strings) + '\n'
        return overview_string

    def get_correct_id_string(self, max_distance, landmark_indizes=None):
        """
        Return the correct id rate string.
        :param max_distance: Distance to check.
        :param landmark_indizes: Indizes of landmarks, on which the measure should be calculated. If None, use all landmarks.
        :return: Overview string.
        """
        num_correct_id = self.get_num_correct_id(max_distance, landmark_indizes)
        _, num_valid_gt, _ = self.get_num_valid(landmark_indizes)
        return ('#correct_id <= {}: {} out of {} ({' + self.float_format_string + '}%)\n').format(max_distance, num_correct_id, num_valid_gt, 100 * num_correct_id / num_valid_gt if num_valid_gt > 0 else 0)

    def get_per_landmark_string(self, landmark_index, outlier_radius, correct_id_max_distance):
        """
        Return the stats string for the given landmark index, which describes several landmark stats.
        :param landmark_index: The landmark index.
        :param outlier_radius: The outlier radius.
        :param correct_id_max_distance: The correct_id distance.
        :return: The stats string for the landmark.
        """
        stats = self.get_pe_statistics([landmark_index])
        _, num_valid_gt, _ = self.get_num_valid([landmark_index])
        num_missed, num_too_many = self.get_num_missed_landmarks([landmark_index])
        landmark_string = f'L{landmark_index:02} (#{num_valid_gt: >2}): {stats[0]: >7.3f} +/- {stats[1]: >7.3f}, {stats[2]: >7.3f}, #miss: {num_missed: >2}'
        if outlier_radius is not None:
            num_outliers = self.get_num_outliers([outlier_radius], False, [landmark_index])[0]
            landmark_string += f', o_r{outlier_radius}: {num_outliers: >2}'
        if correct_id_max_distance is not None:
            num_correct_id = self.get_num_correct_id(correct_id_max_distance, [landmark_index])
            landmark_string += f', #not_id_r{correct_id_max_distance}: {num_valid_gt - num_correct_id}'
        return landmark_string

    def get_landmarks_string(self, outlier_radius, correct_id_max_distance):
        """
        Return the stats string for all landmarks.
        :param outlier_radius: The outlier radius.
        :param correct_id_max_distance: The correct_id distance.
        :return: The stat string for all landmarks.
        """
        num_landmarks = len(list(self.distances.values())[0])
        landmarks_strings = []
        for i in range(num_landmarks):
            landmarks_strings.append(self.get_per_landmark_string(i, outlier_radius, correct_id_max_distance))
        return '\n'.join(landmarks_strings) + '\n'

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
        if per_instance_outliers_radius is not None or correct_id_distance is not None:
            overview_string += self.get_per_instance_string(per_instance_outliers_radius, correct_id_distance)
        if correct_id_distance is not None:
            overview_string += self.get_correct_id_string(correct_id_distance)
        overview_string += self.get_landmarks_string(outlier_radii[-1], correct_id_distance)
        missed, too_many = self.get_num_missed_landmarks()
        overview_string += f'missed landmarks: {missed}; too many: {too_many}\n'
        return overview_string
