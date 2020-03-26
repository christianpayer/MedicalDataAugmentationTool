
import SimpleITK as sitk
import numpy as np
import utils.geometry
import utils.sitk_image
import utils.sitk_np
import utils.np_image
import utils.io.image
import utils.io.text
import utils.io.common


class SegmentationTest(object):
    """
    Creates the predicted labels for multi label segmentation tasks.
    """
    def __init__(self,
                 labels,
                 channel_axis,
                 interpolator='linear',
                 largest_connected_component=False,
                 all_labels_are_connected=False):
        """
        Initializer.
        :param labels: The list of labels to generate. Usually something like list(range(num_labels)).
        :param channel_axis: The channel axis of the numpy array that corresponds to the label probabilities.
        :param interpolator: The interpolator to use for resampling the numpy predictions.
        :param largest_connected_component: If true, filter the labels such that only the largest connected component per labels gets returned.
        :param all_labels_are_connected: If true, filter labels such that all labels are connected.
        """
        self.labels = labels
        self.channel_axis = channel_axis
        self.interpolator = interpolator
        self.largest_connected_component = largest_connected_component
        self.all_labels_are_connected = all_labels_are_connected
        self.metric_values = {}

    def get_transformed_image(self, prediction_np, reference_sitk=None, output_spacing=None, transformation=None):
        """
        Returns the transformed predictions as a list of sitk images. If the transformation is None, the prediction_np image
        will not be transformed, but only split and converted to a list of sitk images.
        :param prediction_np: The predicted np array.
        :param reference_sitk: The reference sitk image from which origin/spacing/direction is taken from.
        :param output_spacing: The output spacing of the prediction_np array.
        :param transformation: The sitk transformation used to transform the reference_sitk image to the network input.
        :return: A list of the transformed sitk predictions.
        """
        if transformation is not None:
            predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction_np,
                                                                                  output_spacing=output_spacing,
                                                                                  channel_axis=self.channel_axis,
                                                                                  input_image_sitk=reference_sitk,
                                                                                  transform=transformation,
                                                                                  interpolator=self.interpolator,
                                                                                  output_pixel_type=sitk.sitkFloat32)
            prediction_np = utils.sitk_np.sitk_list_to_np(predictions_sitk, axis=0)
        else:
            if self.channel_axis != 0:
                prediction_np = utils.np_image.split_by_axis(prediction_np, self.channel_axis)
                prediction_np = np.stack(prediction_np, axis=0)
        return prediction_np

    def get_prediction_labels_list(self, prediction):
        """
        Converts network predictions to the predicted labels.
        :param prediction: The network predictions as np array.
        :return: List of the predicted labels as np arrays.
        """
        num_labels = len(self.labels)
        prediction_labels = utils.np_image.argmax(prediction, axis=0)
        return utils.np_image.split_label_image(prediction_labels, list(range(num_labels)))

    def get_predictions_labels(self, prediction):
        """
        Converts a list of sitk network predictions to the sitk label image.
        Also performs postprocessing, see postprocess_prediction_labels.
        :param predictions_list_sitk: A list of sitk images.
        :return: The predicted labels as an sitk image.
        """
        prediction = self.postprocess_prediction_labels(prediction)
        prediction_labels = utils.np_image.argmax(prediction, axis=0)
        for label_index, target_label in zip(range(len(self.labels)), self.labels):
            if label_index != target_label:
                prediction_labels[prediction_labels == label_index] = target_label
        return prediction_labels

    def postprocess_prediction_labels(self, prediction):
        """
        Postprocesses np network predictions, see filter_all_labels_are_connected and filter_largest_connected_component.
        :param prediction: The np network predictions.
        :return: The postprocessed np network predictions.
        """
        if self.all_labels_are_connected:
            self.filter_all_labels_are_connected(prediction)
        if self.largest_connected_component:
            self.filter_largest_connected_component(prediction)
        return prediction

    def filter_largest_connected_component(self, prediction):
        """
        Filters the predictions such that only the largest connected component per label remains.
        :param prediction: The np network predictions.
        """
        while True:
            prediction_labels_list = self.get_prediction_labels_list(prediction)
            prediction_labels = np.stack(prediction_labels_list, axis=0)
            prediction_labels_largest_cc_list = [utils.np_image.largest_connected_component(prediction_labels)
                                                 for prediction_labels in prediction_labels_list[1:]]
            prediction_labels_largest_cc = np.stack([prediction_labels_list[0]] + prediction_labels_largest_cc_list, axis=0)
            # filter pixels that are in the prediction labels but not in the largest cc
            prediction_filter = prediction_labels != prediction_labels_largest_cc
            # break if no pixels would be filtered
            if not np.any(prediction_filter):
                break
            prediction[prediction_filter] = -np.inf

    def filter_all_labels_are_connected(self, prediction):
        """
        Filters the predictions such that all predicted labels are connected.
        :param prediction: The np network predictions.
        """
        # split into background and other labels
        prediction_background, prediction_others = np.split(prediction, [1], axis=0)
        # remove unused dimension in background
        prediction_background = np.squeeze(prediction_background, axis=0)
        # merge other labels by using the max among all labels
        prediction_others = np.max(prediction_others, axis=0)
        # stack background and merged labels
        prediction_background_others = np.stack([prediction_background, prediction_others], axis=0)
        # find arg max -> either background or other labels
        all_labels_prediction = utils.np_image.argmax(prediction_background_others, axis=0)
        # get largest component of other labels
        all_labels_prediction = utils.np_image.largest_connected_component(all_labels_prediction)
        # filter is the largest component
        prediction_filter = np.stack([all_labels_prediction] * prediction.shape[0], axis=0) == 0
        prediction[prediction_filter] = -np.inf

    def get_label_image(self, prediction_np, reference_sitk=None, output_spacing=None, transformation=None, return_transformed_sitk=False, image_type=None):
        """
        Returns the label image as an sitk image. Performs resampling and postprocessing.
        :param prediction_np: The np network predictions.
        :param reference_sitk: The reference sitk image from which origin/spacing/direction is taken from.
        :param output_spacing: The output spacing of the prediction_np array.
        :param transformation: The sitk transformation used to transform the reference_sitk image to the network input.
        :param return_transformed_sitk: If true, also return the transformed predictions as sitk images.
        :param image_type: The image type to convert the np labels to, before converting them to an sitk image.
        :return: The predicted labels as an sitk image.
        """
        assert len(self.labels) == prediction_np.shape[self.channel_axis], 'number of labels must be equal to prediction image channel axis'
        prediction_transformed = self.get_transformed_image(prediction_np, reference_sitk, output_spacing, transformation)
        prediction_labels = self.get_predictions_labels(prediction_transformed)
        prediction_labels_sitk = utils.sitk_np.np_to_sitk(prediction_labels, type=image_type)
        if reference_sitk is not None:
            prediction_labels_sitk.CopyInformation(reference_sitk)
        if return_transformed_sitk:
            return prediction_labels_sitk, prediction_transformed
        else:
            return prediction_labels_sitk

